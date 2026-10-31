"""Within-pair AUC of the attempt-length effort score on the 24 pilot pairs, per attempter and for means of attempters.
Effort = -log(1 + output tokens) (a mean over attempters for a combo such as luna+haiku). Item outcome = mean of the pair's
responses on the item, binarized at 0.5. Only items that every listed attempter answered are used, so all rows share one
item set. AUC per pair (pairs with both outcomes), then the mean over pairs.
Usage (run from lab/): attempt_auc.py luna,haiku,sonnet,opus,luna+haiku"""
import os as _os, pathlib as _pathlib  # project root: $PAIEC_ROOT, else derived from this file's location
_ROOT = _os.environ.get("PAIEC_ROOT") or str(_pathlib.Path(__file__).resolve().parents[1])
import sys
LAB = _ROOT + "/lab"
CODE = _ROOT + "/eval"
sys.path.insert(0, LAB); sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from h21 import K
CL = {"haiku": "cl_haiku_med", "sonnet": "cl_sonnet_med", "opus": "cl_opus_med"}


def claude_total(name):
    d = pd.read_parquet(f"{LAB}/features/{CL[name]}.parquet")
    return {(b, i): -np.log1p(o) for b, i, o in zip(d.benchmark, d.item_id, d.output_tokens) if np.isfinite(o)}


combos = [c.split("+") for c in sys.argv[1].split(",")]
st = K(); ti = st.features.index("total")
src = {"luna": {k: v[ti] for k, v in st.F.items()}}
for n in {n for c in combos for n in c if n != "luna"}:
    src[n] = claude_total(n)
items = set.intersection(*(set(src[n]) for c in combos for n in c))
y = st.m.groupby(["benchmark", "subject", "item_id"]).y.mean().reset_index()
y = y[[(b, i) in items for b, i in zip(y.benchmark, y.item_id)]].copy()
y["yb"] = (y.y >= 0.5).astype(int)
print(f"pilot pairs {y.groupby(['benchmark', 'subject']).ngroups}, items answered by every listed attempter {len(y)}")
for c in combos:
    y["e"] = [float(np.mean([src[n][(b, i)] for n in c])) for b, i in zip(y.benchmark, y.item_id)]
    a = [roc_auc_score(g.yb, g.e) for _, g in y.groupby(["benchmark", "subject"]) if g.yb.nunique() == 2]   # e is high for short attempts
    print(f"{'+'.join(c):>12}: within-pair AUC {np.mean(a):.4f} over {len(a)} pairs")
