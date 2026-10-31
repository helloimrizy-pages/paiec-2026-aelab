#!/usr/bin/env python3
"""Build items.parquet / results.parquet for LiveCodeBench Pro.

Usage:  python -I build.py <api_snapshot.json.gz> <hf_parquet_dir> <out_dir> [--dry-run]

<api_snapshot.json.gz>  snapshot of the leaderboard backend https://webhook.cp-bench.orzzh.com (not versioned;
                        the shipped file was retrieved 2026-10-19T18:47Z; fetch_api.py makes a new one):
                        {"llm", "difficulty", "submissions", "fetch_log"}
<hf_parquet_dir>        the six split parquet files (data/*.parquet) of the gated dataset
                        https://huggingface.co/datasets/QAQAQAQAQ/LiveCodeBench-Pro
                        @ adebffce047dddb7768a86bace6aea4f7425e3bc. Access needs a one-time
                        "request access" on Hugging Face (automatic approval).
--dry-run               compute and print results counts without the parquet files and without writing.

Outcome rule (reproduces the leaderboard pass rate exactly):
  accepted -> 1.0 ; not_accepted, compilation_error, generation_failed -> 0.0 ;
  unknown, in_batch, in_queue -> no row (no grade).
"""
import collections, glob, gzip, json, os, sys
import pandas as pd

snap_path, hf_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
dry = "--dry-run" in sys.argv
snap = json.loads(gzip.decompress(open(snap_path, "rb").read()))
GRADE = {"accepted": 1.0, "not_accepted": 0.0, "compilation_error": 0.0, "generation_failed": 0.0}
NOGRADE = {"unknown", "in_batch", "in_queue"}

llm = {}
for m in snap["llm"]:          # (name, provider) is the key the backend uses; one pair is listed twice
    k = (m["name"], m["provider"])
    if k not in llm or (m.get("display_name") and m.get("rating")):
        llm[k] = m
disp = collections.Counter(m.get("display_name") for m in llm.values() if m.get("display_name"))


def model_label(name, prov):
    d = llm.get((name, prov), {}).get("display_name")
    if not d:
        return name
    return d if disp[d] == 1 else f"{d} [{name}]"   # e.g. four different runs are all shown as "Stealth"
recs = {}
for key, D in snap["submissions"].items():
    name, prov, mode, ps, diff = key.split("__")
    for c in D["contests"]:
        for p in c["problems"]:
            pid = p["problem_link"].rstrip("/").split("/")[-2] + p["problem_index"]
            v = p["verdict"]
            if v in NOGRADE:
                continue
            if v not in GRADE:
                raise SystemExit(f"unexpected verdict {v}")
            k = (name, prov, pid, p["submission_id"])
            r = recs.setdefault(k, dict(name=name, provider=prov, problem_id=pid, submission_id=p["submission_id"], verdict=v,
                                        status=p.get("status"), contest_title=c["contest_title"], contest_start_time=c["contest_start_time"],
                                        views=set()))
            if r["verdict"] != v:
                raise SystemExit(f"verdict conflict for {k}")
            r["views"].add(f"{mode}:{ps}:{diff}")

rows = []
for (name, prov, pid, sid), r in recs.items():
    m = llm.get((name, prov), {})
    rows.append(dict(problem_id=pid, model=model_label(name, prov), api_name=name, provider=prov,
                     flags="manual_eval" if prov in ("manual_eval", "manual") else "",
                     config=f"provider={prov} | api_name={name} | harness=LiveCodeBench-Pro leaderboard (C++ solution, judged on full tests)",
                     outcome=GRADE[r["verdict"]], verdict=r["verdict"], status=r["status"], submission_id=sid,
                     leaderboard_views=",".join(sorted(r["views"])), organization=m.get("organization"), model_status=m.get("status")))
res = pd.DataFrame(rows)
print("graded (model, problem) rows:", len(res), "models:", res.model.nunique(), "api names:", res.api_name.nunique(),
      "problems:", res.problem_id.nunique())
if dry:
    sys.exit(0)

files = sorted(glob.glob(os.path.join(hf_dir, "*.parquet")))
if len(files) != 6:
    raise SystemExit("need the 6 gated parquet files (data/*.parquet) of QAQAQAQAQ/LiveCodeBench-Pro")
probs = {}
for f in files:
    split = os.path.basename(f).split("-")[0]
    d = pd.read_parquet(f)
    for _, x in d.iterrows():
        e = probs.setdefault(x["problem_id"], dict(row=x, splits=[]))
        e["splits"].append(split)
items = pd.DataFrame([dict(item_key=pid, text=e["row"]["problem_statement"], extra=json.dumps(dict(
    problem_title=e["row"]["problem_title"], difficulty=e["row"]["difficulty"], platform=e["row"]["platform"], link=e["row"]["link"],
    time_limit=int(e["row"]["time_limit"]), memory_limit=int(e["row"]["memory_limit"]), splits=sorted(e["splits"]),
    prompt_prefix="fixed instruction from api_interface.py is prepended to the statement (same for all items)"), ensure_ascii=False))
    for pid, e in probs.items()]).sort_values("item_key")
res = res.rename(columns={"problem_id": "item_key"})
no_text = res[~res.item_key.isin(set(items.item_key))]
res = res[res.item_key.isin(set(items.item_key))].copy()
res = res.sort_values(["model", "config", "item_key", "submission_id"])
res["trial"] = res.groupby(["model", "config", "item_key"]).cumcount() + 1
res = res[["item_key", "model", "config", "trial", "outcome", "flags", "verdict", "status", "submission_id", "api_name", "provider",
           "leaderboard_views", "organization", "model_status"]]
os.makedirs(out, exist_ok=True)
items.to_parquet(os.path.join(out, "items.parquet"), index=False)
res.to_parquet(os.path.join(out, "results.parquet"), index=False)
print(json.dumps(dict(n_items=len(items), n_results=len(res), n_models=res.model.nunique(),
                      dropped_rows_without_text=len(no_text), dropped_problems_without_text=no_text.item_key.nunique()), indent=1))
