"""H15 zero-mean adaptive extra slopes on v5. Usage: extras.py SEED SPLITS TAG variants"""
import os as _os, pathlib as _pathlib  # project root: $PAIEC_ROOT, else derived from this file's location
_ROOT = _os.environ.get("PAIEC_ROOT") or str(_pathlib.Path(__file__).resolve().parents[1])
import sys, json
sys.path.insert(0, _ROOT + "/lab")
import numpy as np, pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, logsumexp
import sim
from ranking_model import RankingResearch, BUDG, boot_delta, posterior_grid
from predictors import base_item
OUT = _ROOT + "/eval/results/"
VARS = {"v5": [], "X_rel": [("rel", .3, 7)], "X_input": [("input", .3, 7)], "X_answer": [("answer", .3, 7)],
        "X_all": [("rel", .3, 5), ("input", .3, 5), ("answer", .3, 5)]}

class X(RankingResearch):
    def __init__(self):
        super().__init__()
        m = self.m
        m["reldev"] = m.rel - m.groupby(["subject", "benchmark"]).rel.transform("mean")
        self.sd = {n: float(m[n + "dev"].std()) for n in ["rel", "input", "answer"]}
        self.gcache = {}

    def sig(self, ep, items, name):
        if name == "rel":
            return np.array([self.v.J[(ep.subject, ep.benchmark, base_item(i))][1] for i in items])
        return self.signals(ep.benchmark, items, [name])[:, 0]

    def grid(self, extras):
        key = tuple(extras)
        if key not in self.gcache:
            U, V, E, LW = posterior_grid(self.nodes, self.slope_nodes, .4, effort_nodes=self.effort_nodes)
            axes = [(U, V, E, LW)]
            Xs = []
            cols = [U, V, E]; lw = LW
            for name, s, nn in extras:
                x, w = hermgauss(nn); x, w = s * np.sqrt(2) * x, np.log(w / np.sqrt(np.pi))
                cols = [np.repeat(c, nn) for c in cols] + []
                Xs = [np.repeat(c, nn) for c in Xs] + [np.tile(x, len(lw))]
                lw = (lw[:, None] + w[None, :]).ravel()
            self.gcache[key] = (cols[0], cols[1], cols[2], Xs, lw)
        return self.gcache[key]

    def pred(self, H, ep, k, var):
        if k == 0 or not VARS[var]:
            return self.predict(H, "adaptive_total", ep, k)
        ex = VARS[var]
        hc, hw, c, coefs = self.fit(H, "adaptive_total")
        li, y = ep.label_items[:k], ep.label_y[:k]
        zt = np.array([self.v.J[(ep.subject, ep.benchmark, base_item(i))][0] for i in ep.target_items])
        zl = np.array([self.v.z(ep.subject, ep.benchmark, i) for i in li])
        ft, fl = self.signals(ep.benchmark, ep.target_items, ["total"])[:, 0], self.signals(ep.benchmark, li, ["total"])[:, 0]
        zmu, zbt = zl.mean(), (zt + zl.sum()) / (k + 1)
        dl, dt = fl - fl.mean(), ft - (ft + fl.sum()) / (k + 1)
        bB, bW, bF = coefs
        ml = c + bB * zmu + bW * (zl - zmu) + dl * bF; mt = c + bB * zbt + bW * (zt - zbt) + dt * bF
        U, V, E, Xs, LW = self.grid(ex)
        fl_ = ml[:, None] + U + (zl - zmu)[:, None] * V + dl[:, None] * E
        xt_list = []
        for (name, s, nn), Xg in zip(ex, Xs):
            sl_, st_ = self.sig(ep, li, name), self.sig(ep, ep.target_items, name)
            xl = (sl_ - sl_.mean()) / self.sd[name]; xt = (st_ - (st_ + sl_.sum()) / (k + 1)) / self.sd[name]
            fl_ = fl_ + xl[:, None] * Xg[None, :]; xt_list.append((xt, Xg))
        lw = LW + (y[:, None] * fl_ - np.logaddexp(0, fl_)).sum(0)
        keep = lw > lw.max() - 25
        post = np.exp(lw[keep] - logsumexp(lw[keep]))
        out = np.empty(len(zt))
        for s0 in range(0, len(zt), 8):
            sl = slice(s0, s0 + 8)
            f = mt[sl, None] + U[keep] + (zt[sl] - zbt[sl])[:, None] * V[keep] + dt[sl, None] * E[keep]
            for xt, Xg in xt_list: f = f + xt[sl, None] * Xg[keep]
            out[sl] = expit(f) @ post
        return out

if __name__ == "__main__":
    seed, splits, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    st = X(); eps = st.episodes(splits, seed); pairs = {}
    for n in sys.argv[4].split(","):
        rows = []
        for ep in eps:
            sc = [float((p * p - 2 * p * ep.ymean + ep.ymean) @ ep.row_weights) for p in (st.pred(ep.benchmark, ep, k, n) for k in sim.BUDGETS)]
            rows.append(dict(benchmark=ep.benchmark, subject=ep.subject, episode=ep.episode, **dict(zip(BUDG, sc)), alc=float(np.dot(sc, sim.WEIGHTS))))
        pp = pd.DataFrame(rows).groupby(["benchmark", "subject"])[BUDG + ["alc"]].mean().reset_index(); pairs[n] = pp
        pp.to_csv(OUT + f"{tag}_{n}_pairs.csv", index=False)
        cv = pp.groupby("benchmark")[BUDG + ["alc"]].mean()
        print(n, round(cv.alc.mean(), 5), json.dumps(boot_delta(pp, pairs["v5"])) if n != "v5" else "", cv.alc.round(4).to_dict(), cv.mean().round(4).to_dict(), flush=True)
