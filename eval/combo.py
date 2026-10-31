"""v7 candidate: v5 + metadata-group shift (H10) + zero-mean adaptive answer slope (H15). Usage: combo.py SEED SPLITS TAG variants"""
import os as _os, pathlib as _pathlib  # project root: $PAIEC_ROOT, else derived from this file's location
_ROOT = _os.environ.get("PAIEC_ROOT") or str(_pathlib.Path(__file__).resolve().parents[1])
import sys, json, re
sys.path.insert(0, _ROOT + "/lab")
sys.path.insert(0, _ROOT + "/eval")
import numpy as np, pandas as pd
from scipy.special import expit, logsumexp
import sim
from ranking_model import BUDG, boot_delta
from predictors import base_item
from extras import X
from groups import tokens_of
from data import D
from v3sim import TEST_LIKE
OUT = _ROOT + "/eval/results/"
VARS = {"v5": ([], 0.), "G": ([], 1.), "A": ([("answer", .3, 7)], 0.), "GA": ([("answer", .3, 7)], 1.), "GA.75": ([("answer", .3, 7)], .75)}

class C(X):
    def __init__(self):
        super().__init__()
        self.T = {}
        for b in TEST_LIKE:
            it = pd.read_parquet(D / b / "items.parquet", columns=["item_id", "item_features"])
            self.T.update({(b, i): tokens_of(f) for i, f in zip(it.item_id, it.item_features)})

    def pred_c(self, H, ep, k, var):
        ex, sg = VARS[var]
        if k == 0 or (not ex and not sg):
            return self.predict(H, "adaptive_total", ep, k)
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
        xts = []
        for (name, s, nn), Xg in zip(ex, Xs):
            sl_, st_ = self.sig(ep, li, name), self.sig(ep, ep.target_items, name)
            fl_ = fl_ + ((sl_ - sl_.mean()) / self.sd[name])[:, None] * Xg[None, :]
            xts.append(((st_ - (st_ + sl_.sum()) / (k + 1)) / self.sd[name], Xg))
        lw = LW + (y[:, None] * fl_ - np.logaddexp(0, fl_)).sum(0)
        keep = lw > lw.max() - 25; post = np.exp(lw[keep] - logsumexp(lw[keep]))
        shift = np.zeros(len(zt))
        if sg:
            pl = expit(fl_[:, keep]) @ post; r, h = y - pl, pl * (1 - pl)
            tl = [self.T.get((ep.benchmark, base_item(i)), frozenset()) for i in li]
            tt = [self.T.get((ep.benchmark, base_item(i)), frozenset()) for i in ep.target_items]
            # Per call: a token is uninformative if the target and every revealed item share it.
            labshared = set.intersection(*[set(x) for x in tl])
            stats = {}
            for toks, ri, hi in zip(tl, r, h):
                for t in toks:
                    a = stats.setdefault(t, [0., 0.]); a[0] += ri; a[1] += hi
            for j, toks in enumerate(tt):
                tk = [t for t in toks if t not in labshared]
                if tk:
                    s2 = sg ** 2 / len(tk)
                    shift[j] = sum(stats[t][0] / (stats[t][1] + 1 / s2) for t in tk if t in stats)
        out = np.empty(len(zt))
        for s0 in range(0, len(zt), 8):
            sl = slice(s0, s0 + 8)
            f = mt[sl, None] + shift[sl, None] + U[keep] + (zt[sl] - zbt[sl])[:, None] * V[keep] + dt[sl, None] * E[keep]
            for xt, Xg in xts: f = f + xt[sl, None] * Xg[keep]
            out[sl] = expit(f) @ post
        return out

if __name__ == "__main__":
    seed, splits, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    st = C(); eps = st.episodes(splits, seed); pairs = {}
    for n in sys.argv[4].split(","):
        rows = []
        for ep in eps:
            sc = [float((p * p - 2 * p * ep.ymean + ep.ymean) @ ep.row_weights) for p in (st.pred_c(ep.benchmark, ep, k, n) for k in sim.BUDGETS)]
            rows.append(dict(benchmark=ep.benchmark, subject=ep.subject, episode=ep.episode, **dict(zip(BUDG, sc)), alc=float(np.dot(sc, sim.WEIGHTS))))
        pp = pd.DataFrame(rows).groupby(["benchmark", "subject"])[BUDG + ["alc"]].mean().reset_index(); pairs[n] = pp
        pp.to_csv(OUT + f"{tag}_{n}_pairs.csv", index=False)
        cv = pp.groupby("benchmark")[BUDG + ["alc"]].mean()
        print(n, round(cv.alc.mean(), 5), json.dumps(boot_delta(pp, pairs["v5"])) if n != "v5" else "", cv.alc.round(4).to_dict(), cv.mean().round(4).to_dict(), flush=True)
