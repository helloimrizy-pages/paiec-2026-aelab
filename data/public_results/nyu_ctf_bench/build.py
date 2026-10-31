"""Build NYU CTF Bench per-item results: items.parquet, results.parquet, submissions.csv.

Usage: python -I build.py <download_dir> <out_dir>
Sources (pinned commits), checked out under <download_dir>:
  leaderboard_submissions  https://github.com/NYU-LLM-CTF/leaderboard_submissions @ 69def02a79e8b1d890e1e0d5c11106a9f40e11e1
  NYU_CTF_Bench            https://github.com/NYU-LLM-CTF/NYU_CTF_Bench @ 4c5744f55a31a58e03d6118c9dbb1c18e5bbdfa9
Outcome: 1.0 if transcripts/<submission>/summary.json has results[<challenge>] true, 0.0 if false.
Challenges missing from a summary get no row.
"""
import json, sys, glob, os, collections
import pandas as pd
D, OUT = sys.argv[1], sys.argv[2]
LB = f"{D}/leaderboard_submissions"; BENCH = f"{D}/NYU_CTF_Bench"
ds = json.load(open(f"{BENCH}/test_dataset.json"))

# ---- items
rows = []
for key, meta in ds.items():
    c = json.load(open(f"{BENCH}/{meta['path']}/challenge.json"))
    points = c.get("points", c.get("initial"))
    extra = {
        "name": c["name"], "category": c["category"], "category_short": key.split("-")[1],
        "points": points, "points_field": "points" if "points" in c else ("initial" if "initial" in c else None),
        "year": meta["year"], "event": meta["event"], "path": meta["path"],
        "files": c.get("files", []), "box": c.get("box"), "internal_port": c.get("internal_port"),
        "author": c.get("author"), "reference": c.get("reference"),
    }
    rows.append({"item_key": key, "text": c.get("description", ""), "extra": json.dumps(extra, ensure_ascii=False)})
items = pd.DataFrame(rows)
assert items.item_key.is_unique and len(items) == 200
items.to_parquet(f"{OUT}/items.parquet", index=False)

# ---- results
res = []; subs = []
for p in sorted(glob.glob(f"{LB}/transcripts/*/summary.json")):
    folder = os.path.basename(os.path.dirname(p))
    s = json.load(open(p)); m = s["metadata"]; r = s["results"]
    config = f"{m['agent']} | {m['comment']}" if m.get("comment") else m["agent"]
    n_ok = 0
    for k, v in r.items():
        if k not in ds or not isinstance(v, bool):
            continue  # not an explicit grade for a known challenge
        res.append({"item_key": k, "model": m["model"], "config": config, "trial": 1, "outcome": 1.0 if v else 0.0})
        n_ok += 1
    subs.append({"folder": folder, "agent": m["agent"], "comment": m["comment"], "model": m["model"],
                 "date": m["date"], "link": m["link"], "n_graded": n_ok, "n_missing": len(set(ds) - set(r)),
                 "n_solved": sum(1 for k, v in r.items() if v is True and k in ds)})
results = pd.DataFrame(res).astype({"trial": "int64", "outcome": "float64"})
assert not results.duplicated(["item_key", "model", "config", "trial"]).any()
results.to_parquet(f"{OUT}/results.parquet", index=False)
pd.DataFrame(subs).to_csv(f"{OUT}/submissions.csv", index=False)
print(json.dumps(subs, indent=1))
print("items", len(items), "empty text", (items.text.str.strip() == "").sum())
print("results", len(results), "submissions", len(subs), "models", results.model.nunique())
print(results.groupby(["model", "config"]).outcome.agg(["count", "sum"]))
print(items.extra.apply(lambda e: json.loads(e)["category"]).value_counts().to_dict())
print(items.extra.apply(lambda e: json.loads(e)["points"] is None).sum(), "items without points")
