"""H22 simulation on the 24 pilot pairs: value of a public-results lookup in three coverage regimes vs v8 (blend50).
Library for a pilot benchmark = its measurement-db responses (all subjects). Configs = subject_ids; model = normalized_name.
  R1 exact: target subject's own config present (harness-matched);  R2 model-only: own config removed, same-model configs kept;
  R3 easiness: every config of the target's model removed.   Usage (from lab/): python ../eval/h22_sim.py SEED SPLITS
"""
import os as _os, pathlib as _pathlib  # project root: $PAIEC_ROOT, else derived from this file's location
_ROOT = _os.environ.get("PAIEC_ROOT") or str(_pathlib.Path(__file__).resolve().parents[1])
import sys, json
LAB = _ROOT + "/lab"
CODE = _ROOT + "/eval"
sys.path.insert(0, LAB); sys.path.insert(0, CODE)
import numpy as np, pandas as pd
import sim
from ranking_model import BUDG, boot_delta
from h21 import K
from h22_mix import priors, mix_predict, easy_predict, shrink
from data import load_long, D
from v3sim import TEST_LIKE
RES = _ROOT + "/eval/results/"

st = K(); st.mode = "blend50"
seed, splits = int(sys.argv[1]), int(sys.argv[2])
eps = st.episodes(splits, seed)
df = load_long(TEST_LIKE)
names = pd.concat([pd.read_parquet(D / b / "subjects.parquet")[["subject_id", "normalized_name"]] for b in TEST_LIKE]).drop_duplicates("subject_id").set_index("subject_id").normalized_name
Y = {}
for b, g in df.groupby("benchmark"):
    t = g.groupby(["item", "subject"]).y.mean()
    d = {}
    for (it, sj), v in t.items(): d.setdefault(it, {})[sj] = float(v)
    Y[b] = d
SUBJ = {b: g.subject.unique() for b, g in df.groupby("benchmark")}

def lib_view(b, target, regime):
    model = names.get(target)
    same = [s for s in SUBJ[b] if model is not None and names.get(s) == model]
    if regime == "R1": cfgs = same if target in same else [target]
    elif regime == "R2": cfgs = [s for s in same if s != target]
    else: cfgs = []
    return cfgs, set(same) | {target}

_im = {}
def item_mean(b, item, excl, key):
    k = (b, item, key)
    if k not in _im:
        y = Y[b].get(item, {}); v = [x for sj, x in y.items() if sj not in excl]
        _im[k] = shrink(sum(v), len(v), a=1.0, prior=0.5) if v else None
    return _im[k]

rows = []
for ep in eps:
    b, s = ep.benchmark, ep.subject
    base = {k: st.pred_c(b, ep, k, "GA") for k in sim.BUDGETS}
    for regime in ["R1", "R2", "R3"]:
        cfgs, excl = lib_view(b, s, regime); key = (s, regime)
        w, w0 = priors([c == s for c in cfgs], len(cfgs))
        sc = []
        for k in sim.BUDGETS:
            li, ly = ep.label_items[:k], ep.label_y[:k]
            lab = [(y2, [Y[b].get(it2, {}).get(c) for c in cfgs], item_mean(b, it2, excl, key)) for it2, y2 in zip(li, ly)]
            lp = [(y2, m2) for y2, _, m2 in lab if m2 is not None]
            preds = []
            for j, it in enumerate(ep.target_items):
                tgt = [Y[b].get(it, {}).get(c) for c in cfgs]
                m_t = item_mean(b, it, excl, key)
                p_fb = float(base[k][j]) if m_t is None else easy_predict(m_t, lp)   # fallback = easiness when the item is known
                if cfgs and any(o is not None for o in tgt):
                    p, _ = mix_predict(tgt, w, w0, p_fb, lab)
                else:
                    p = p_fb
                preds.append(p)
            p = np.asarray(preds)
            sc.append(float((p * p - 2 * p * ep.ymean + ep.ymean) @ ep.row_weights))
        rows.append(dict(regime=regime, benchmark=b, subject=s, episode=ep.episode, n_cfg=len(cfgs), **dict(zip(BUDG, sc)), alc=float(np.dot(sc, sim.WEIGHTS))))
    sc = [float((base[k] ** 2 - 2 * base[k] * ep.ymean + ep.ymean) @ ep.row_weights) for k in sim.BUDGETS]
    rows.append(dict(regime="v8", benchmark=b, subject=s, episode=ep.episode, n_cfg=0, **dict(zip(BUDG, sc)), alc=float(np.dot(sc, sim.WEIGHTS))))
R = pd.DataFrame(rows); R.to_csv(RES + f"h22_sim_easyfb_s{seed}.csv", index=False)
P = R.groupby(["regime", "benchmark", "subject"])[BUDG + ["alc", "n_cfg"]].mean().reset_index()
print("pairs per regime with >=1 same-model config (R2):", int((P[P.regime == "R2"].n_cfg > 0).sum()), "of", int((P.regime == "R2").sum()))
for g in ["v8", "R1", "R2", "R3"]:
    q = P[P.regime == g]; cv = q.groupby("benchmark")[BUDG + ["alc"]].mean()
    d = "" if g == "v8" else json.dumps(boot_delta(q[["benchmark", "subject", "alc"]], P[P.regime == "v8"][["benchmark", "subject", "alc"]]))
    print(g, round(cv.alc.mean(), 4), cv.alc.round(4).to_dict(), cv.mean()[BUDG].round(4).to_dict(), d, flush=True)
q = P[(P.regime == "R2") & (P.n_cfg > 0)]; v = P[P.regime == "v8"].merge(q[["benchmark", "subject"]])
print("R2 restricted to pairs that have same-model configs:", round(q.alc.mean(), 4), "vs v8", round(v.alc.mean(), 4))
