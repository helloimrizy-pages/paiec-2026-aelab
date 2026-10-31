"""H10 metadata-group adjustment on v5. Usage: groups.py SEED SPLITS TAG sg1,sg2,..."""
import os as _os, pathlib as _pathlib  # project root: $PAIEC_ROOT, else derived from this file's location
_ROOT = _os.environ.get("PAIEC_ROOT") or str(_pathlib.Path(__file__).resolve().parents[1])
import sys, json, re
sys.path.insert(0, _ROOT + "/lab")
import numpy as np, pandas as pd
from scipy.special import expit, logsumexp
import sim
from data import D
from ranking_model import RankingResearch, BUDG, boot_delta, posterior_grid
from predictors import base_item
from v3sim import TEST_LIKE
OUT = _ROOT + "/eval/results/"

def tokens_of(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return frozenset()
    out = []
    for part in re.split(r"[;\n]", str(s)):
        if "=" not in part: continue
        k, v = part.split("=", 1); k = k.strip().lower()
        if "idx" in k or k.endswith("id") or k in ("prompt_source",): continue
        out.append(f"{k}={v.strip()}")
    return frozenset(out)

class G(RankingResearch):
    def __init__(self):
        super().__init__()
        self.T = {}
        for b in TEST_LIKE:
            it = pd.read_parquet(D / b / "items.parquet", columns=["item_id", "item_features"])
            self.T.update({(b, i): tokens_of(f) for i, f in zip(it.item_id, it.item_features)})

    def predict_g(self, H, ep, budget, sg):
        if budget == 0 or sg == 0:
            return self.predict(H, "adaptive_total", ep, budget)
        hc, hw, c, coefs = self.fit(H, "adaptive_total")
        li, y = ep.label_items[:budget], ep.label_y[:budget]
        jt = np.array([self.v.J[(ep.subject, ep.benchmark, base_item(i))] for i in ep.target_items])
        zt = jt[:, 0]; zl = np.array([self.v.z(ep.subject, ep.benchmark, i) for i in li])
        ft, fl = self.signals(ep.benchmark, ep.target_items, ["total"]), self.signals(ep.benchmark, li, ["total"])
        n = len(zl); zmu, zbt = zl.mean(), (zt + zl.sum()) / (n + 1)
        dl, dt = fl - fl.mean(0), ft - (ft + fl.sum(0)) / (n + 1)
        bB, bW, bF = coefs
        ml = c + bB * zmu + bW * (zl - zmu) + dl[:, 0] * bF
        mt = c + bB * zbt + bW * (zt - zbt) + dt[:, 0] * bF
        if (.4, False) not in self.grids:
            self.grids[(.4, False)] = posterior_grid(self.nodes, self.slope_nodes, .4, effort_nodes=self.effort_nodes)
        U, V, E, LW = self.grids[(.4, False)]
        ll = ml[:, None] + U + (zl - zmu)[:, None] * V + dl[:, 0, None] * E
        lw = LW + (y[:, None] * ll - np.logaddexp(0, ll)).sum(0); post = np.exp(lw - logsumexp(lw))
        pl = expit(ll) @ post
        r, h = y - pl, pl * (1 - pl)
        tl = [self.T.get((ep.benchmark, base_item(i)), frozenset()) for i in li]
        tt = [self.T.get((ep.benchmark, base_item(i)), frozenset()) for i in ep.target_items]
        allvis = set.intersection(*[set(x) for x in tl + tt]) if tl else set()
        stats = {}
        for toks, ri, hi in zip(tl, r, h):
            for t in toks:
                if t in allvis: continue
                a = stats.setdefault(t, [0., 0.]); a[0] += ri; a[1] += hi
        shift = np.zeros(len(tt))
        for j, toks in enumerate(tt):
            tk = [t for t in toks if t not in allvis]
            if not tk: continue
            s2 = sg ** 2 / len(tk)
            shift[j] = sum(stats[t][0] / (stats[t][1] + 1 / s2) for t in tk if t in stats)
        out = np.empty(len(zt))
        for st in range(0, len(zt), 24):
            sl = slice(st, st + 24)
            out[sl] = expit(mt[sl, None] + shift[sl, None] + U + (zt[sl] - zbt[sl])[:, None] * V + dt[sl, 0, None] * E) @ post
        return out

def run(st, eps, sg):
    rows = []
    for ep in eps:
        sc = [float((p * p - 2 * p * ep.ymean + ep.ymean) @ ep.row_weights) for p in (st.predict_g(ep.benchmark, ep, k, sg) for k in sim.BUDGETS)]
        rows.append(dict(benchmark=ep.benchmark, subject=ep.subject, episode=ep.episode, **dict(zip(BUDG, sc)), alc=float(np.dot(sc, sim.WEIGHTS))))
    return pd.DataFrame(rows).groupby(["benchmark", "subject"])[BUDG + ["alc"]].mean().reset_index()

if __name__ == "__main__":
    seed, splits, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    sgs = [float(x) for x in sys.argv[4].split(",")]
    st = G(); eps = st.episodes(splits, seed)
    base = run(st, eps, 0); print("v5", round(base.groupby("benchmark").alc.mean().mean(), 5), flush=True)
    for sg in sgs:
        pp = run(st, eps, sg); pp.to_csv(OUT + f"{tag}_sg{sg}_pairs.csv", index=False)
        cv = pp.groupby("benchmark")[BUDG + ["alc"]].mean()
        print(sg, round(cv.alc.mean(), 5), json.dumps(boot_delta(pp, base)), cv.alc.round(4).to_dict(), cv.mean().round(4).to_dict(), flush=True)
