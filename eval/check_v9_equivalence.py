"""Offline v9 deployment checks (blend50 + two-attempter effort + public library) with cached public data; never calls inference APIs."""
import os as _os, pathlib as _pathlib  # project root: $PAIEC_ROOT, else derived from this file's location
_ROOT = _os.environ.get("PAIEC_ROOT") or str(_pathlib.Path(__file__).resolve().parents[1])
import importlib.util, json, sys, time
from pathlib import Path
sys.path.insert(0, _ROOT + "/lab")
sys.path.insert(0, _ROOT + "/eval")
import numpy as np, pandas as pd
CODE_DIR = Path(_ROOT + "/eval")
from combo import C
from data import D
from v3sim import TEST_LIKE
SRC = Path(_ROOT + "/submissions/v9_lib")

def load():
    sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location("v9_sub", SRC / "model.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

_src = open(_ROOT + "/eval/h23_ens.py").read().split("seed, splits, tag")[0]
exec(_src)
research = Ens(); research.mode = "blend50"
_luna_total = {k: v[research.features.index("total")] for k, v in research.F.items()}
_haiku_total = claude_total("haiku")
research.set_combo(["luna", "haiku"])
_cal = research.fit(None, "adaptive_total")
research.cal[(None, "adaptive_total", "blend50")] = _cal
_cfg = json.load(open(SRC / "judge_config.json"))
assert abs(_cal[0] - _cfg["head_c"]) < 1e-12 and np.allclose(_cal[1], _cfg["head_w"]) and abs(_cal[2] - _cfg["mund_c"]) < 1e-12 and np.allclose(_cal[3], [_cfg["mund_bB"], _cfg["mund_bW"], _cfg["mund_bE"]]), "config constants differ from refit"
print("v9 constants match judge_config.json", flush=True)
FEAT = {}
for b in TEST_LIKE:
    it = pd.read_parquet(D / b / "items.parquet", columns=["item_id", "item_features"])
    FEAT.update({(b, i): f for i, f in zip(it.item_id, it.item_features)})
sub = load(); sub.USE_CACHE = False; sub.LIB = None
ti, ai = research.features.index("total"), research.features.index("answer")
sub._judge = lambda s, it, d: dict(zip(["z", "rel", "t"], research.v.J[(s["_sid"], it["benchmark_id"], it["_iid"])]))
def att(it, d):
    k = (it["benchmark_id"], it["_iid"]); ot = float(np.expm1(-_luna_total[k])); ans = float(np.expm1(-research.F[k][ai]))
    return {"ot": ot, "rt": ot - ans}
def att2(it, d):
    k = (it["benchmark_id"], it["_iid"])
    return {"ot": float(np.expm1(-_haiku_total[k]))} if k in _haiku_total else None
sub._attempt = att; sub._attempt2 = att2

def input_for(s, b, item):
    iid = item.split("|")[0]
    subject = {k: "" for k in ["normalized_name", "provider", "release_date", "access_date", "harness", "reasoning_effort", "harness_version", "subject_features_extra"]}
    subject.update(normalized_name=s, _sid=s)
    f = FEAT[(b, iid)]
    return [subject, dict(item_content="cached-public-item " + iid, item_features=None if not isinstance(f, str) else f, interactors="", benchmark_id=b, _iid=iid)]

t0 = time.monotonic(); worst, calls = 0., 0
for ep in research.episodes(1, 71203):
    for k in [0, 1, 3, 7, 15, 31]:
        exp = research.pred_c(None, ep, k, "GA")
        if k == 0:
            from numpy.polynomial.hermite import hermgauss
            from scipy.special import expit
            gx, gw = hermgauss(21); gx, gw = np.sqrt(2) * gx, gw / np.sqrt(np.pi)
            hc, hw = research.fit(None, "adaptive_total")[:2]
            jt = np.array([research.v.J[(ep.subject, ep.benchmark, i.split("|")[0])] for i in ep.target_items])
            exp = expit((hc + jt @ hw)[:, None] + 1.25 * gx[None, :]) @ gw
        labeled = [[input_for(ep.subject, ep.benchmark, i), int(y)] for i, y in zip(ep.label_items[:k], ep.label_y[:k])]
        if k:
            foreign = input_for(ep.subject, ep.benchmark, ep.label_items[0]); foreign[0] = {**foreign[0], "normalized_name": "unrelated", "_sid": "unrelated"}
            labeled.append([foreign, 1])
        for j in [0, len(ep.target_items) // 2, len(ep.target_items) - 1]:
            a = sub.predict(input_for(ep.subject, ep.benchmark, ep.target_items[j]), labeled)
            r = float(np.clip(exp[j], sub.P_MIN, sub.P_MAX)); worst = max(worst, abs(a - r)); calls += 1
            if abs(a - r) > 1e-9: raise AssertionError(f"mismatch {ep.benchmark} k={k}: {a} vs {r}")
    print("ok", ep.benchmark, ep.subject, calls, flush=True)
# sibling evidence: a sibling success on the same item raises p, a failure lowers it; other items do nothing
ep = research.episodes(1, 71203)[0]
tgt = input_for(ep.subject, ep.benchmark, ep.target_items[0])
own = [[input_for(ep.subject, ep.benchmark, i), int(y)] for i, y in zip(ep.label_items[:7], ep.label_y[:7])]
def sib(item, y):
    x = input_for(ep.subject, ep.benchmark, item); x[0] = {**x[0], "normalized_name": "sib", "_sid": "sib"}; return [x, y]
base = sub.predict(tgt, own)
up = sub.predict(tgt, own + [sib(ep.target_items[0], 1), sib(ep.label_items[1], 0)])
down = sub.predict(tgt, own + [sib(ep.target_items[0], 0), sib(ep.label_items[1], 1)])
same = sub.predict(tgt, own + [sib(ep.label_items[1], 1)])
assert up > base > down and abs(same - base) < 1e-12, (base, up, down, same)
# fault paths
sub2 = load(); sub2.USE_CACHE = False; sub2.LIB = None
sub2._judge = lambda *a: None; sub2._attempt = lambda *a: None; sub2._attempt2 = lambda *a: None
for labels in [[], [[tgt, 0]], [[tgt, 1]] * 31]:
    g = sub2.predict(tgt, labels); assert isinstance(g, float) and 0 <= g <= 1
sub3 = load(); sub3.USE_CACHE = False; sub3.LIB = None
sub3._judge = sub._judge; sub3._attempt = lambda it, d: {"ot": 8000.0, "rt": 8000.0, "timeout": True}; sub3._attempt2 = lambda it, d: None
g = sub3.predict(tgt, own); assert 0 <= g <= 1
import h22_library as _devlib
_have_lib = (_devlib.LIB / "sh_hash.npy").exists() and any((_devlib.SRC / x / "items.parquet").exists() for x in ([l.strip() for l in open(_devlib.SRC / "READY.txt")] if (_devlib.SRC / "READY.txt").exists() else []))
lib_checks, p_exact, row = "skipped (library not built; see README.md)", None, {"mean": None}
if _have_lib:
    # library path: matcher identical to development, mixture code identical file, item-only and exact-config behaviour
    import hashlib as _hl, h22_library as dev
    assert _hl.sha256(open(SRC / "h22_mix.py", "rb").read()).digest() == _hl.sha256(open(CODE_DIR / "h22_mix.py", "rb").read()).digest(), "h22_mix.py differs"
    lib_dev = dev.Library(); sub4 = load(); assert sub4.LIB is not None, "library not loaded"
    lib_checks = 0
    for slug in sorted(set(lib_dev.items.slug)):
        it = pd.read_parquet(dev.SRC / slug / "items.parquet")
        for r in it.sample(min(40, len(it)), random_state=1).itertuples():
            wrapped = json.dumps({"prompt": "Task:\n" + str(r.text), "id": "z"})
            a, b = lib_dev.match(wrapped), sub4.LIB.match(wrapped)
            assert (a[0] if a else None) == b, (slug, r.item_key, a, b); lib_checks += 1
    O = pd.read_parquet(dev.LIB / "outcomes.parquet"); row = O[O.canon.notna()].iloc[0]
    itx = pd.read_parquet(dev.SRC / row.slug / "items.parquet").set_index("item_key").loc[row.item_key, "text"]
    subj = {k: "" for k in ["normalized_name", "provider", "release_date", "access_date", "harness", "reasoning_effort", "harness_version", "subject_features_extra"]}
    subj.update(normalized_name=row.canon, harness=str(row.config).split("|")[0].strip())
    tgt = [subj, dict(item_content=json.dumps({"prompt": str(itx)}), item_features="", interactors="", benchmark_id="bench_x")]
    sub4._predict = lambda *a: (_ for _ in ()).throw(AssertionError("library hit must not call the v8 path"))
    p_exact = sub4.predict(tgt, []); assert abs(p_exact - row["mean"]) < 0.3, (p_exact, row["mean"])
    lib_checks += 1
out = dict(status="passed", library_match_checks=lib_checks, library_exact_hit_p=p_exact, library_exact_hit_recorded=(None if row["mean"] is None else float(row["mean"])), cached_prediction_checks=calls, max_abs_difference=worst, sibling=dict(base=base, up=up, down=down),
           seconds=time.monotonic() - t0, network_calls=0)
Path(_ROOT + "/eval/results/v9_deployment_checks.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out))
