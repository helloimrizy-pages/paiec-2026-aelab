"""H23: v8 (blend50) with the effort feature 'total' = mean over attempter models of -log(1 + output tokens).
Usage (run from lab/): h23_ens.py SEED SPLITS TAG combos   e.g. combos = luna,luna+haiku,luna+haiku+opus"""
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
RES = _ROOT + "/eval/results/"
CL = {"haiku": "cl_haiku_med", "sonnet": "cl_sonnet_med", "opus": "cl_opus_med"}

def claude_total(name):
    d = pd.read_parquet(f"{LAB}/features/{CL[name]}.parquet")
    return {(b, i): -np.log1p(o) for b, i, o in zip(d.benchmark, d.item_id, d.output_tokens) if np.isfinite(o)}

class Ens(K):
    def set_combo(self, combo):
        ti = self.features.index("total")
        if not hasattr(self, "_F0"): self._F0 = dict(self.F); self._m0 = self.m[["total", "totaldev"]].copy()
        srcs = [{k: v[ti] for k, v in self._F0.items()}] if "luna" in combo else []
        srcs += [claude_total(n) for n in combo if n != "luna"]
        F = {}
        for k, v in self._F0.items():
            vals = [s[k] for s in srcs if k in s]
            v = list(v); v[ti] = float(np.mean(vals)) if vals else v[ti]; F[k] = tuple(v)
        self.F = F
        self.m["total"] = [F[(b, i)][ti] for b, i in zip(self.m.benchmark, self.m.item_id)]
        self.m["totaldev"] = self.m.total - self.m.groupby(["subject", "benchmark"]).total.transform("mean")
        self.cal = {}

seed, splits, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
st = Ens(); st.mode = "blend50"; eps = st.episodes(splits, seed); out = {}
for combo in sys.argv[4].split(","):
    st.set_combo(combo.split("+"))
    rows = []
    for ep in eps:
        sc = [float((p * p - 2 * p * ep.ymean + ep.ymean) @ ep.row_weights) for p in (st.pred_c(ep.benchmark, ep, k, "GA") for k in sim.BUDGETS)]
        rows.append(dict(benchmark=ep.benchmark, subject=ep.subject, episode=ep.episode, **dict(zip(BUDG, sc)), alc=float(np.dot(sc, sim.WEIGHTS))))
    pp = pd.DataFrame(rows).groupby(["benchmark", "subject"])[BUDG + ["alc"]].mean().reset_index(); out[combo] = pp
    pp.to_csv(RES + f"{tag}_{combo}_pairs.csv", index=False)
    cv = pp.groupby("benchmark")[BUDG + ["alc"]].mean()
    first = list(out)[0]
    print(combo, round(cv.alc.mean(), 5), json.dumps(boot_delta(pp, out[first])) if combo != first else "", cv.alc.round(4).to_dict(), flush=True)
