"""H21: v7 with the budget-0 head and/or the pair-level mapping calibrated on the 97-benchmark checkpoint judgments.
Usage: h21.py pilot SEED SPLITS TAG MODES      gate test on the 4 pilot benchmarks (each held out of the pilot part of the fit)
       h21.py diag                             LOBO diagnostics over the checkpoint benchmarks (level error by benchmark)
MODES (comma separated): pilot (= v7 control) | ckpt_head | ckpt_level | ckpt_both | ckpt_lw | both_both
  ckpt_*: fitted on the checkpoint pairs only; both_*: checkpoint + pilot pairs (minus the held-out benchmark).
  head = budget-0 logistic head on [z, rel, t]; level = Mundlak intercept and between-pair slope (c, bB);
  lw = also the within-pair judge slope bW from the checkpoint. The effort slope always stays from the pilot fit.
"""
import os as _os, pathlib as _pathlib  # project root: $PAIEC_ROOT, else derived from this file's location
_ROOT = _os.environ.get("PAIEC_ROOT") or str(_pathlib.Path(__file__).resolve().parents[1])
import sys, json
LAB = _ROOT + "/lab"
CODE = _ROOT + "/eval"
sys.path.insert(0, LAB); sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
import sim
from ranking_model import BUDG, boot_delta
from h18 import E
from v3sim import TEST_LIKE, REL
RES = _ROOT + "/eval/results/"
CKPT = LAB + "/features/apijudge_mdb_level.parquet"


def LR():
    return LogisticRegression(C=100, max_iter=2000)


def ck_rows(path=CKPT):
    """Checkpoint judgments -> one weighted row per (pair, item, outcome); each pair has total weight 1."""
    d = pd.read_parquet(path)
    d["z"] = logit(np.clip(d.p_success, .01, .99)); d["rel"] = d.relative_difficulty.map(REL).astype(float)
    d["t"] = logit(np.clip(d.typical_success_rate, .01, .99))
    g = d.groupby(["benchmark", "subject"])
    d["zbar"] = g.z.transform("mean"); d["zdev"] = d.z - d.zbar; d["tbar"] = g.t.transform("mean"); d["relbar"] = g.rel.transform("mean")
    yb = (d.item_y_raw >= 0.5).astype(float) if "binarize" in sys.argv else d.item_y_raw.astype(float)
    x = pd.concat([d.assign(y=1.0, w=yb), d.assign(y=0.0, w=1 - yb)])
    x = x[x.w > 0].copy()
    x["w"] = x.w / x.groupby(["benchmark", "subject"]).w.transform("sum")
    return x


class K(E):
    mode = "pilot"

    def __init__(self):
        super().__init__()
        import os
        self.ck = ck_rows() if os.path.exists(CKPT) else None
        if self.ck is not None: print("checkpoint rows", len(self.ck), "pairs", self.ck.groupby(["benchmark", "subject"]).ngroups, "benchmarks", self.ck.benchmark.nunique(), flush=True)

    def fit(self, H, variant):
        key = (H, variant, self.mode)
        if key in self.cal:
            return self.cal[key]
        m = self.m
        tr = m[m.benchmark != H]
        w = 1 / tr.groupby(["subject", "benchmark"]).y.transform("size")
        head = LR().fit(tr[["z", "rel", "t"]], tr.y, sample_weight=w)
        f = LR().fit(tr[["zbar", "zdev", "totaldev"]], tr.y, sample_weight=w)
        hc, hw, c, coefs = float(head.intercept_[0]), head.coef_[0].copy(), float(f.intercept_[0]), f.coef_[0].copy()
        if self.mode != "pilot":
            x = self.ck
            if self.mode.startswith("both"):
                pil = tr[["benchmark", "subject", "z", "rel", "t", "zbar", "zdev", "y"]].assign(w=w.to_numpy())
                x = pd.concat([x[pil.columns], pil], ignore_index=True)
            h2 = LR().fit(x[["z", "rel", "t"]], x.y, sample_weight=x.w)
            l2 = LR().fit(x[["zbar", "zdev"]], x.y, sample_weight=x.w)
            hc2, hw2, c2, bB2, bW2 = float(h2.intercept_[0]), h2.coef_[0].copy(), float(l2.intercept_[0]), float(l2.coef_[0][0]), float(l2.coef_[0][1])
            if self.mode in ("ckpt_head",):
                hc, hw = hc2, hw2
            elif self.mode in ("ckpt_level",):
                c, coefs = c2, np.array([bB2, coefs[1], coefs[2]])
            elif self.mode in ("ckpt_both", "both_both"):
                hc, hw, c, coefs = hc2, hw2, c2, np.array([bB2, coefs[1], coefs[2]])
            elif self.mode == "ckpt_lw":
                hc, hw, c, coefs = hc2, hw2, c2, np.array([bB2, bW2, coefs[2]])
            elif self.mode.startswith("blend"):
                lam = int(self.mode[5:]) / 100
                hc, hw = (1 - lam) * hc + lam * hc2, (1 - lam) * hw + lam * hw2
                c, coefs = (1 - lam) * c + lam * c2, np.array([(1 - lam) * coefs[0] + lam * bB2, coefs[1], coefs[2]])
            else:
                raise SystemExit(f"unknown mode {self.mode}")
        self.cal[key] = (hc, hw, c, coefs)
        return self.cal[key]


def score(st, eps):
    rows = []
    for ep in eps:
        sc = [float((p * p - 2 * p * ep.ymean + ep.ymean) @ ep.row_weights) for p in (st.pred_c(ep.benchmark, ep, k, "GA") for k in sim.BUDGETS)]
        rows.append(dict(benchmark=ep.benchmark, subject=ep.subject, episode=ep.episode, **dict(zip(BUDG, sc)), alc=float(np.dot(sc, sim.WEIGHTS))))
    return pd.DataFrame(rows).groupby(["benchmark", "subject"])[BUDG + ["alc"]].mean().reset_index()


def diag():
    """Leave-one-checkpoint-benchmark-out: budget-0 Brier and pair-level error, deployed v7 constants vs checkpoint fit."""
    cfg = json.load(open(_ROOT + "/eval/results/v7_constants.json"))
    x = ck_rows()
    prof = pd.read_csv(_ROOT + "/data/mdb_profile.csv").set_index("benchmark")
    rows = []
    for B in sorted(x.benchmark.unique()):
        tr, te = x[x.benchmark != B], x[x.benchmark == B]
        h2 = LR().fit(tr[["z", "rel", "t"]], tr.y, sample_weight=tr.w); l2 = LR().fit(tr[["zbar", "zdev"]], tr.y, sample_weight=tr.w)
        for (b, s), g in te.groupby(["benchmark", "subject"]):
            it = g.drop_duplicates(["item_id"])
            yb = it.item_y_raw.to_numpy(float); rate = float(g.pair_rate_bin.iloc[0])
            X = it[["z", "rel", "t"]].to_numpy()
            p_dep = expit(cfg["head_c"] + X @ np.asarray(cfg["head_w"])); p_new = expit(h2.intercept_[0] + X @ h2.coef_[0])
            lv_dep = float(expit(cfg["mund_c"] + cfg["mund_bB"] * it.zbar.iloc[0])); lv_new = float(expit(l2.intercept_[0] + l2.coef_[0][0] * it.zbar.iloc[0]))
            rows.append(dict(benchmark=b, subject=s, rate=rate, judge_mean_p=float(expit(it.z).mean()), recent=bool(g.recent.iloc[0]),
                             b0_dep=float(np.mean(p_dep ** 2 - 2 * p_dep * yb + yb)), b0_new=float(np.mean(p_new ** 2 - 2 * p_new * yb + yb)),
                             lvl_dep=lv_dep, lvl_new=lv_new, modality=str(prof.modality.get(b, "")), domain=str(prof.domain.get(b, ""))))
    D = pd.DataFrame(rows)
    D["b0_const"] = 0.25  # always-0.5 reference
    D["lvl_err_dep"] = D.lvl_dep - D.rate; D["lvl_err_new"] = D.lvl_new - D.rate
    D.to_csv(RES + "h21_diag_pairs.csv", index=False)
    print("pairs", len(D), "| b0 Brier: deployed v7 head", round(D.b0_dep.mean(), 4), "| checkpoint-LOBO head", round(D.b0_new.mean(), 4))
    print("level error (prob scale) mean signed / abs: deployed", round(D.lvl_err_dep.mean(), 3), round(D.lvl_err_dep.abs().mean(), 3),
          "| checkpoint-LOBO", round(D.lvl_err_new.mean(), 3), round(D.lvl_err_new.abs().mean(), 3))
    D["agentic"] = D.domain.str.contains("agents_and_tool_use|software_engineering")
    print(D.groupby("agentic")[["rate", "judge_mean_p", "lvl_err_dep", "lvl_err_new", "b0_dep", "b0_new"]].mean().round(3).to_string())
    print(D.groupby("recent")[["rate", "judge_mean_p", "lvl_err_dep", "lvl_err_new", "b0_dep", "b0_new"]].mean().round(3).to_string())
    bb = D.groupby("benchmark")[["rate", "judge_mean_p", "lvl_err_dep", "lvl_err_new", "b0_dep", "b0_new"]].mean()
    print(bb.sort_values("lvl_err_dep").round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "diag":
        diag(); sys.exit()
    seed, splits, tag = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    st = K(); eps = st.episodes(splits, seed); out = {}
    for mode in sys.argv[5].split(","):
        st.mode = mode; pp = score(st, eps); out[mode] = pp
        pp.to_csv(RES + f"{tag}_{mode}_pairs.csv", index=False)
        cv = pp.groupby("benchmark")[BUDG + ["alc"]].mean()
        print(mode, round(cv.alc.mean(), 5), json.dumps(boot_delta(pp, out["pilot"])) if "pilot" in out and mode != "pilot" else "",
              cv.alc.round(4).to_dict(), cv.mean().round(4).to_dict(), flush=True)
