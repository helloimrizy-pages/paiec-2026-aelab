"""Judge vs labels only on the 291 checkpoint pairs (97 benchmarks), with the final model form.
Same short episodes and seeds as h21c.py (15 acquisition + 15 evaluation items, budgets 0,1,3,7,15, 10 splits per pair).
Judge part of the final model (no attempt lengths, which are not cached for these items):
  logit P = c + bB*zbar + bW*(z - zbar) + u + v*(z - zbar),  u ~ 0.7 N(0, 0.5^2) + 0.3 N(0, 2^2),  v ~ N(0, 0.6^2),
  posterior by quadrature; budget 0 uses the head on [z, rel, t] (tau = 0, as in all simulations).
Labels-only baseline: logit P = c0 + u with the same prior on u.
Calibrations: judge_pilot (v7 constants, fitted on the 4 pilot benchmarks), judge_ckpt_lobo (fitted on the other 96
checkpoint benchmarks), judge_blend_lobo (50/50 average of the two, as in v8/v9, with the target benchmark held out),
judge_final_insample (the constants of the submitted program). c0 is fitted on the other 96 benchmarks (labels_lobo_c0)
or is the program's fallback constant (labels_pilot_c0). Partial ALC = (0.1 b0 + 0.2 b1 + 0.2 b3 + 0.2 b7 + 0.2 b15) / 0.9.
Usage (from lab/): python ../eval/transfer_check.py
"""
import os as _os, pathlib as _pathlib  # project root: $PAIEC_ROOT, else derived from this file's location
_ROOT = _os.environ.get("PAIEC_ROOT") or str(_pathlib.Path(__file__).resolve().parents[1])
import sys, json
LAB = _ROOT + "/lab"
sys.path.insert(0, LAB)
import numpy as np, pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit, logit, log_expit
from sklearn.linear_model import LogisticRegression
from v3sim import REL

RES = _ROOT + "/eval/results/"
BUD = [0, 1, 3, 7, 15]; W = np.array([.1, .2, .2, .2, .2]) / 0.9
v7 = json.load(open(RES + "v7_constants.json"))
fin = json.load(open(_ROOT + "/submissions/v9_lib/judge_config.json"))
FALLBACK_C0 = -0.573   # constant prior logit of the program's label-only fallback

def gh(n, sd):
    x, w = hermegauss(n); return x * sd, w / w.sum()
u1, wu1 = gh(41, 0.5); u2, wu2 = gh(41, 2.0)
U = np.r_[u1, u2]; WU = np.r_[0.7 * wu1, 0.3 * wu2]
V, WV = gh(21, 0.6)
UG, VG = np.meshgrid(U, V, indexing="ij"); LP = np.log(np.outer(WU, WV)); LP0 = np.log(WU)

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
    return dict(hc=float(h.intercept_[0]), hw=h.coef_[0].copy(), c=float(l.intercept_[0]), bB=float(l.coef_[0][0]), bW=float(l.coef_[0][1]))

ck = prep(pd.read_parquet(LAB + "/features/apijudge_mdb_level.parquet")); ckw = wrows(ck, "item_y_raw")
pilot = dict(hc=v7["head_c"], hw=np.asarray(v7["head_w"]), c=v7["mund_c"], bB=v7["mund_bB"], bW=v7["mund_bW"])
final = dict(hc=fin["head_c"], hw=np.asarray(fin["head_w"]), c=fin["mund_c"], bB=fin["mund_bB"], bW=fin["mund_bW"])

def judge_ep(g, c, rng):
    idx = rng.permutation(len(g)); acq, ev = idx[:15], idx[15:30]
    z, rel, t, y = g.z.to_numpy(), g.rel.to_numpy(), g.t.to_numpy(), g.item_y_raw.to_numpy()
    zt, yt = z[ev], y[ev]; out = []
    for k in BUD:
        if k == 0:
            p = expit(c["hc"] + np.c_[z[ev], rel[ev], t[ev]] @ c["hw"])
        else:
            li = acq[:k]; zl, yl = z[li], (y[li] >= 0.5).astype(float)
            zb_l = zl.mean(); zb_t = (zt + zl.sum()) / (k + 1)
            ml = c["c"] + c["bB"] * zb_l + c["bW"] * (zl - zb_l); dl = zl - zb_l
            f = ml[:, None, None] + UG[None] + VG[None] * dl[:, None, None]
            ll = (yl[:, None, None] * log_expit(f) + (1 - yl[:, None, None]) * log_expit(-f)).sum(0) + LP
            w = np.exp(ll - ll.max()); w /= w.sum()
            mt = c["c"] + c["bB"] * zb_t + c["bW"] * (zt - zb_t); dt = zt - zb_t
            p = (expit(mt[:, None, None] + UG[None] + VG[None] * dt[:, None, None]) * w[None]).sum((1, 2))
        out.append(float(np.mean(p * p - 2 * p * yt + yt)))
    return out

def labels_ep(g, c0, rng):
    idx = rng.permutation(len(g)); acq, ev = idx[:15], idx[15:30]
    y = g.item_y_raw.to_numpy(); yt = y[ev]; out = []
    for k in BUD:
        if k == 0:
            p0 = float((expit(c0 + U) * WU).sum())
        else:
            yl = (y[acq[:k]] >= 0.5).astype(float); f = c0 + U
            ll = yl.sum() * log_expit(f) + (k - yl.sum()) * log_expit(-f) + LP0
            w = np.exp(ll - ll.max()); w /= w.sum(); p0 = float((expit(f) * w).sum())
        p = np.full(len(yt), p0); out.append(float(np.mean(p * p - 2 * p * yt + yt)))
    return out

rows = []
for B in sorted(ck.benchmark.unique()):
    other = ckw[ckw.benchmark != B]; q = float(np.average(other.y, weights=other.w))
    kf = fit(other)
    blend = dict(hc=.5 * pilot["hc"] + .5 * kf["hc"], hw=.5 * pilot["hw"] + .5 * kf["hw"], c=.5 * pilot["c"] + .5 * kf["c"], bB=.5 * pilot["bB"] + .5 * kf["bB"], bW=pilot["bW"])
    for s, g in ck[ck.benchmark == B].groupby("subject"):
        if len(g) < 30: continue
        seeds = [int(r.integers(2**31)) for r in [np.random.default_rng(7 + i) for i in range(10)]]
        variants = [("labels_lobo_c0", lambda r: labels_ep(g, float(logit(q)), r)), ("labels_pilot_c0", lambda r: labels_ep(g, FALLBACK_C0, r)),
                    ("judge_pilot", lambda r: judge_ep(g, pilot, r)), ("judge_ckpt_lobo", lambda r: judge_ep(g, kf, r)),
                    ("judge_blend_lobo", lambda r: judge_ep(g, blend, r)), ("judge_final_insample", lambda r: judge_ep(g, final, r))]
        for name, fn in variants:
            sc = np.mean([fn(np.random.default_rng(sd)) for sd in seeds], axis=0)
            rows.append(dict(cal=name, benchmark=B, subject=s, **{f"b{k}": v for k, v in zip(BUD, sc)}, palc=float(sc @ W)))
D = pd.DataFrame(rows); D.to_csv(RES + "transfer_check_pairs.csv", index=False)
cols = [f"b{k}" for k in BUD] + ["palc"]
print("pairs:", (D.cal == "judge_pilot").sum(), "benchmarks:", D.benchmark.nunique())
print("mean over pairs:")
print(D.groupby("cal")[cols].mean().round(4).to_string())

def boot(a, b, n=4000):
    """mean over benchmarks of the per-benchmark mean change; pairs resampled within each benchmark"""
    d = a.merge(b, on=["benchmark", "subject"], suffixes=("", "_c")); groups = [(g.palc - g.palc_c).to_numpy() for _, g in d.groupby("benchmark")]
    rng = np.random.default_rng(0); bs = np.mean([rng.choice(x, (n, len(x))).mean(1) for x in groups], axis=0)
    pdl = (d.palc - d.palc_c).to_numpy()
    return dict(delta=round(float(np.mean([x.mean() for x in groups])), 4), ci95=np.percentile(bs, [2.5, 97.5]).round(4).tolist(),
                benchmarks_improved=f"{int(sum(x.mean() < 0 for x in groups))}/{len(groups)}", pairs_improved=f"{int((pdl < 0).sum())}/{len(pdl)}")
for j in ["judge_pilot", "judge_ckpt_lobo", "judge_blend_lobo", "judge_final_insample"]:
    for l in ["labels_lobo_c0", "labels_pilot_c0"]:
        print(f"{j:22s} vs {l:16s}", boot(D[D.cal == j], D[D.cal == l]))
