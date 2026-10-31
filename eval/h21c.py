"""H21c: mini-episodes on the 291 checkpoint pairs (15 acquisition + 15 eval items, budgets 0,1,3,7,15), judge+GP model
(v3 form: budget-0 head on [z, rel, t]; then GP with mean c + bB*zbar + bW*zdev). Compares calibrations:
  deployed (v7 judge_config constants, fitted on the 4 pilot benchmarks) vs ckpt_lobo (fitted on the other 96 checkpoint
  benchmarks) vs both_lobo (other 96 + the 4 pilot benchmarks). Partial ALC = (0.1 b0 + 0.2 b1 + 0.2 b3 + 0.2 b7 + 0.2 b15) / 0.9."""
import os as _os, pathlib as _pathlib  # project root: $PAIEC_ROOT, else derived from this file's location
_ROOT = _os.environ.get("PAIEC_ROOT") or str(_pathlib.Path(__file__).resolve().parents[1])
import sys, json
LAB = _ROOT + "/lab"
sys.path.insert(0, LAB)
import numpy as np, pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from tta import GPPair, KernelParams
from v3sim import REL, TEST_LIKE
from data import load_long
RES = _ROOT + "/eval/results/"
BUD = [0, 1, 3, 7, 15]; W = np.array([.1, .2, .2, .2, .2]) / 0.9
cfg = json.load(open(_ROOT + "/eval/results/v7_constants.json"))   # constants of submission v7 (fitted on the 4 pilot benchmarks)
kp = KernelParams(0.75, 0.0, 0.0, sz=0.6)

def prep(d):
    d = d.copy()
    d["z"] = logit(np.clip(d.p_success, .01, .99)); d["rel"] = d.relative_difficulty.map(REL).astype(float); d["t"] = logit(np.clip(d.typical_success_rate, .01, .99))
    g = d.groupby(["benchmark", "subject"]); d["zbar"] = g.z.transform("mean"); d["zdev"] = d.z - d.zbar
    return d

def wrows(d, ycol):
    """weighted rows (y in {0,1}) with each pair total weight 1"""
    x = pd.concat([d.assign(y=1.0, w=d[ycol]), d.assign(y=0.0, w=1 - d[ycol])]); x = x[x.w > 0].copy()
    x["w"] = x.w / x.groupby(["benchmark", "subject"]).w.transform("sum"); return x

def fit(x):
    h = LogisticRegression(C=100, max_iter=2000).fit(x[["z", "rel", "t"]], x.y, sample_weight=x.w)
    l = LogisticRegression(C=100, max_iter=2000).fit(x[["zbar", "zdev"]], x.y, sample_weight=x.w)
    return float(h.intercept_[0]), h.coef_[0].copy(), float(l.intercept_[0]), float(l.coef_[0][0]), float(l.coef_[0][1])

ck = prep(pd.read_parquet(LAB + "/features/apijudge_mdb_level.parquet"))
# pilot judgments with true outcomes (per response), for the "both" calibration
pj = prep(pd.read_parquet(LAB + "/features/apijudge_luna_low.parquet"))
pl = load_long(TEST_LIKE); pl["item_id"] = pl["item"].str.split("|").str[0]
pil = pl.merge(pj, on=["subject", "benchmark", "item_id"]); pil = pil.groupby(["benchmark", "subject", "item_id"]).agg(y_raw=("y", "mean"), z=("z", "first"), rel=("rel", "first"), t=("t", "first"), zbar=("zbar", "first"), zdev=("zdev", "first")).reset_index()
ckw, pilw = wrows(ck, "item_y_raw"), wrows(pil, "y_raw")
cal = {"deployed": (cfg["head_c"], np.asarray(cfg["head_w"]), cfg["mund_c"], cfg["mund_bB"], cfg["mund_bW"])}
for B in ck.benchmark.unique():
    cal[("ckpt", B)] = fit(ckw[ckw.benchmark != B]); cal[("both", B)] = fit(pd.concat([ckw[ckw.benchmark != B], pilw], ignore_index=True))
    d, k = cal["deployed"], cal[("ckpt", B)]
    for lam in (25, 50, 75):
        l = lam / 100; cal[(f"blend{lam}", B)] = ((1 - l) * d[0] + l * k[0], (1 - l) * d[1] + l * k[1], (1 - l) * d[2] + l * k[2], (1 - l) * d[3] + l * k[3], d[4])

def episode_scores(g, c, rng):
    hc, hw, mc, bB, bW = c
    idx = rng.permutation(len(g)); acq, ev = idx[:15], idx[15:30]
    z, rel, t, y = g.z.to_numpy(), g.rel.to_numpy(), g.t.to_numpy(), g.item_y_raw.to_numpy()
    zt, yt = z[ev], y[ev]
    out = []
    for k in BUD:
        if k == 0:
            p = expit(hc + np.c_[z[ev], rel[ev], t[ev]] @ hw)
        else:
            li = acq[:k]; zl, yl = z[li], (y[li] >= 0.5).astype(float)
            zb_t = (zt + zl.sum()) / (k + 1); zb_l = zl.mean()
            gp = GPPair([frozenset()] * k, None, yl, mc + bB * zb_l + bW * (zl - zb_l), kp, z=zl - zb_l)
            p = gp.predict([frozenset()] * len(zt), None, mc + bB * zb_t + bW * (zt - zb_t), z=zt - zb_t)
        out.append(float(np.mean(p * p - 2 * p * yt + yt)))
    return out

rows = []
for (B, s), g in ck.groupby(["benchmark", "subject"]):
    if len(g) < 30: continue
    rng = np.random.default_rng(7)
    seeds = [np.random.default_rng(rng.integers(2**31)) for _ in range(10)]
    for name in ["deployed", "ckpt", "both", "blend25", "blend50", "blend75"]:
        c = cal["deployed"] if name == "deployed" else cal[(name, B)]
        sc = np.mean([episode_scores(g, c, np.random.default_rng(int(r.integers(2**31)))) for r in [np.random.default_rng(7 + i) for i in range(10)]], axis=0)
        rows.append(dict(cal=name, benchmark=B, subject=s, **{f"b{k}": v for k, v in zip(BUD, sc)}, palc=float(sc @ W)))
D = pd.DataFrame(rows); D.to_csv(RES + "h21c_pairs.csv", index=False)
cols = [f"b{k}" for k in BUD] + ["palc"]
print("mean over pairs (equal weight), then over benchmarks:")
print(D.groupby("cal")[cols].mean().round(4).to_string())
bench = D.groupby(["cal", "benchmark"])[cols].mean()
print(bench.groupby("cal").mean().round(4).to_string())
def boot(a, b, n=4000):
    d = a.merge(b, on=["benchmark", "subject"], suffixes=("", "_c")); groups = [(g.palc - g.palc_c).to_numpy() for _, g in d.groupby("benchmark")]
    rng = np.random.default_rng(0); bs = np.mean([rng.choice(x, (n, len(x))).mean(1) for x in groups], axis=0)
    return dict(delta=float(np.mean([x.mean() for x in groups])), ci95=np.percentile(bs, [2.5, 97.5]).round(4).tolist(), benchmarks_improved=int(sum(x.mean() < 0 for x in groups)), benchmarks=len(groups))
for name in ["ckpt", "both", "blend25", "blend50", "blend75"]:
    print(name, "vs deployed:", boot(D[D.cal == name], D[D.cal == "deployed"]))
print("deployed vs always-0.5 at b0:", round(D[D.cal == "deployed"].b0.mean(), 4), "vs", round(float(np.mean(0.25)), 4))
