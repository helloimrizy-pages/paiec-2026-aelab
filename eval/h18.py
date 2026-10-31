"""H18: v7 (GA) calibrated with vs without the mmdocrag/swe_rebench pseudo-pairs.
Usage: h18.py pilot SEED SPLITS TAG | h18.py extfolds SEED SPLITS TAG"""
import os as _os, pathlib as _pathlib  # project root: $PAIEC_ROOT, else derived from this file's location
_ROOT = _os.environ.get("PAIEC_ROOT") or str(_pathlib.Path(__file__).resolve().parents[1])
import sys, json
sys.path.insert(0, _ROOT + "/lab")
sys.path.insert(0, _ROOT + "/eval")
import numpy as np, pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
import sim
from data import load_long, D
from ranking_model import BUDG, boot_delta, PairEpisode
from combo import C
from groups import tokens_of
RES = _ROOT + "/eval/results/"
EXT = RES + "ext/"
REL = {"much_easier": 2.0, "easier": 1.0, "typical": 0.0, "harder": -1.0, "much_harder": -2.0}
NEW = ["mmdocrag", "swe_rebench"]

class E(C):
    use_ext = False
    def __init__(self):
        super().__init__()
        p = pd.read_csv(EXT + "pairs.csv", dtype=str).fillna("")
        j = pd.read_parquet(EXT + "judge.parquet"); a = pd.read_parquet(EXT + "attempt.parquet")
        jm = {(r.benchmark, r.source_subject, r.item_id): (float(logit(np.clip(r.p_success, .01, .99))), REL[r.relative_difficulty], float(logit(np.clip(r.typical_success_rate, .01, .99)))) for r in j.itertuples()}
        for r in p.itertuples():
            if (r.benchmark, r.source_subject, r.item_id) in jm: self.v.J[(r.subject, r.benchmark, r.item_id)] = jm[(r.benchmark, r.source_subject, r.item_id)]
        for r in a.itertuples():
            ot, rt = float(r.output_tokens), float(r.reasoning_tokens)
            self.F[(r.benchmark, r.item_id)] = (-np.log1p(rt), -np.log1p(max(ot - rt, 0)), -np.log1p(ot), rt / max(ot, 1), -np.log1p(ot), 0.)
        rows = []
        for b in NEW:
            d = load_long([b]); d["item_id"] = d["item"].str.split("|").str[0]
            q = p[p.benchmark == b]
            if b == "mmdocrag":
                d = d[d["item"].str.endswith("|pure-text/quotes15")].merge(q[["subject", "item_id"]], on=["subject", "item_id"])
            else:
                d = d.drop(columns="subject").merge(q[["subject", "item_id"]], on="item_id")
            rows.append(d)
        x = pd.concat(rows)
        x = x[[((s, b, i) in self.v.J) and ((b, i) in self.F) for s, b, i in zip(x.subject, x.benchmark, x.item_id)]].copy()
        self.extdf = x
        x["z"], x["rel"], x["t"] = zip(*[self.v.J[(s, b, i)] for s, b, i in zip(x.subject, x.benchmark, x.item_id)])
        for k, name in enumerate(self.features):
            x[name] = [self.F[(b, i)][k] for b, i in zip(x.benchmark, x.item_id)]
        g = x.groupby(["subject", "benchmark"])
        x["zbar"] = g.z.transform("mean"); x["zdev"] = x.z - x.zbar
        for name in self.features: x[name + "dev"] = x[name] - g[name].transform("mean")
        self.mext = pd.concat([self.m, x[[c for c in self.m.columns if c in x.columns]]], ignore_index=True)
        for b in NEW:
            it = pd.read_parquet(D / b / "items.parquet", columns=["item_id", "item_features"])
            self.T.update({(b, i): tokens_of(f) for i, f in zip(it.item_id, it.item_features)})
        print("ext rows", len(x), "pairs", x.groupby(["subject", "benchmark"]).ngroups, "rates", x.groupby("benchmark").y.mean().round(3).to_dict(), flush=True)

    def fit(self, H, variant):
        key = (H, variant, self.use_ext)
        if key in self.cal: return self.cal[key]
        m = self.mext if self.use_ext else self.m
        tr = m[m.benchmark != H]
        w = 1 / tr.groupby(["subject", "benchmark"]).y.transform("size")
        head = LogisticRegression(C=100, max_iter=1000).fit(tr[["z", "rel", "t"]], tr.y, sample_weight=w)
        f = LogisticRegression(C=100, max_iter=1000).fit(tr[["zbar", "zdev", "totaldev"]], tr.y, sample_weight=w)
        self.cal[key] = (head.intercept_[0], head.coef_[0], float(f.intercept_[0]), f.coef_[0].copy()); return self.cal[key]

def score(st, eps, ext):
    st.use_ext = ext; rows = []
    for ep in eps:
        sc = [float((p * p - 2 * p * ep.ymean + ep.ymean) @ ep.row_weights) for p in (st.pred_c(ep.benchmark, ep, k, "GA") for k in sim.BUDGETS)]
        rows.append(dict(benchmark=ep.benchmark, subject=ep.subject, episode=ep.episode, **dict(zip(BUDG, sc)), alc=float(np.dot(sc, sim.WEIGHTS))))
    return pd.DataFrame(rows).groupby(["benchmark", "subject"])[BUDG + ["alc"]].mean().reset_index()

def report(a, b, tag):
    a.to_csv(RES + f"{tag}_cal3_pairs.csv", index=False); b.to_csv(RES + f"{tag}_cal5_pairs.csv", index=False)
    ca, cb = a.groupby("benchmark")[BUDG + ["alc"]].mean(), b.groupby("benchmark")[BUDG + ["alc"]].mean()
    print("v7 own calibration ", round(ca.alc.mean(), 5), ca.alc.round(4).to_dict(), ca.mean().round(4).to_dict())
    print("v7 + ext calibration", round(cb.alc.mean(), 5), json.dumps(boot_delta(b, a)), cb.alc.round(4).to_dict(), cb.mean().round(4).to_dict(), flush=True)

def ext_episodes(st, splits, seed):
    x = st.extdf; out = []
    for b in NEW:
        sub = x[x.benchmark == b][["subject", "benchmark", "item", "y"]]; pairs, rng = sim.eligible_pairs(sub), np.random.default_rng(seed)
        for k in range(splits):
            ep = sim.make_episode(sub, pairs, rng)
            for (s, _), t in ep.eval_rows.groupby(["subject", "benchmark"], sort=True):
                ids = t["item"].drop_duplicates().tolist(); g = t.groupby("item").y.agg(["mean", "size"]).loc[ids]
                lab = ep.acq_stream[ep.acq_stream.subject == s].sort_values("rank")
                out.append(PairEpisode(b, s, k, ids, lab["item"].tolist(), lab.y.to_numpy(float), g["mean"].to_numpy(float), g["size"].to_numpy(float) / g["size"].sum()))
    return out

if __name__ == "__main__":
    st = E(); mode = sys.argv[1]
    if mode == "pilot":
        eps = st.episodes(int(sys.argv[3]), int(sys.argv[2])); report(score(st, eps, False), score(st, eps, True), sys.argv[4])
    else:
        st.use_ext = True  # held-out fold H is excluded from fitting automatically (benchmark != H)
        eps = ext_episodes(st, int(sys.argv[3]), int(sys.argv[2])); pp = score(st, eps, True)
        pp.to_csv(RES + f"{sys.argv[4]}_pairs.csv", index=False); print(pp.groupby("benchmark")[BUDG + ["alc"]].mean().round(4).to_string())
