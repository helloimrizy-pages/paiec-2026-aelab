"""Build items.parquet / results.parquet / results_meta.parquet for the Open Proof Corpus (OPC).

Usage: python -I build.py <OPC snapshot dir> <output dir>   (fetch.py downloads the snapshot and runs this)
Source: HF dataset INSAIT-Institute/OPC (https://huggingface.co/datasets/INSAIT-Institute/OPC)
at revision dcc3b4804e2d126ea34b13e3e0cd998c3302644b. The snapshot dir must contain
data/<split>-00000-of-00001.parquet for the six splits.
Outcome = mean of the human 0/1 proof grades in the row's `score` list.
"""
import sys, os, json, glob
import numpy as np
import pandas as pd

SRC, OUT = sys.argv[1], sys.argv[2]
SPLITS = ["generic", "test", "putnambench", "matharena", "pass_at_n", "best_of_n"]  # fixed order -> trial numbering
PLACEHOLDER = "The model was unable to solve this problem."
# release dates of the generating model (from configs/models/*.yaml in github.com/insait-institute/open-proof-corpus @ e92a6ca)
RELEASE = {
    "openai/o3": "2025-04-16", "openai/o4-mini--high": "2025-04-16", "gemini/gemini-pro-2.5": "2025-03-25",
    "qwen/qwen3_235b_a22b": "2025-04-29", "deepseek/r1_0528": "2025-05-28", "xai/grok-3-mini-high": "2025-04-09",
}

frames = []
for s in SPLITS:
    (p,) = glob.glob(f"{SRC}/data/{s}-*.parquet")
    df = pd.read_parquet(p)
    df["hf_split"] = s
    df["row_in_file"] = np.arange(len(df))
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)
assert len(raw) == 4934, len(raw)

# ---------------- items
def clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, str):
        return v if v.strip() else None
    return v

assert raw.groupby("problem_id").problem.nunique().max() == 1
text_to_ids = raw.drop_duplicates("problem_id").groupby("problem").problem_id.apply(sorted).to_dict()
rows = []
for pid, g in raw.groupby("problem_id", sort=True):
    r = g.iloc[0]
    for c in ["competition", "level", "source", "url", "year", "problem_issue"]:
        assert g[c].nunique(dropna=False) == 1, (pid, c)
    cat = r.category
    extra = {
        "competition": r.competition,
        "source": r.source,
        "url": clean(r.url),
        "year": None if clean(r.year) is None else int(r.year),
        "level": r.level,
        "category": None if cat is None else [str(x) for x in cat],
        "problem_issue": clean(r.problem_issue),
        "hf_splits": [s for s in SPLITS if s in set(g.hf_split)],
        "split_field_values": sorted(set(g.split)),
        "has_ground_truth_solution": bool(g.ground_truth_solution.fillna("").str.strip().ne("").any()),
        "same_text_as": [x for x in text_to_ids[r.problem] if x != pid],
    }
    rows.append({"item_key": pid, "text": r.problem, "extra": json.dumps(extra, ensure_ascii=False)})
items = pd.DataFrame(rows)
assert items.item_key.is_unique

# ---------------- results
raw = raw.sort_values(["hf_split", "row_in_file"], key=lambda s: s.map(SPLITS.index) if s.name == "hf_split" else s, kind="stable")
res, meta, dropped = [], [], []
for _, r in raw.iterrows():
    grades = [int(x) for x in r.score]
    judges = [str(x) for x in r.judge_id]
    placeholder = r.solution.startswith(PLACEHOLDER)
    why = None
    if len(grades) == 0 or any(x not in (0, 1) for x in grades):
        why = f"grade not 0/1: {grades}"
    elif placeholder and any(grades):
        why = f"placeholder 'unable to solve' solution but grades {grades}"
    if why:
        dropped.append({"problem_id": r.problem_id, "model_id": r.model_id, "hf_split": r.hf_split, "reason": why})
        continue
    # a judge listed more than once in a row always repeats the same grade (checked), so the mean is unaffected
    for j in set(judges):
        assert len({gr for gr, jj in zip(grades, judges) if jj == j}) == 1, (r.problem_id, r.model_id)
    res.append({"item_key": r.problem_id, "model": r.model_id, "config": f"split={r.hf_split}",
                "outcome": float(np.mean(grades))})
    meta.append({"hf_split": r.hf_split, "split_field": r.split, "n_grades": len(grades),
                 "n_distinct_judges": len(set(judges)), "grades": json.dumps(grades), "judge_ids": json.dumps(judges),
                 "any_uncertain": bool(any(r.uncertain)), "auto_graded": bool(placeholder),
                 "output_tokens": int(r.output_tokens), "cost_usd": float(r.cost), "timestamp": r.timestamp})
results = pd.DataFrame(res)
results["trial"] = results.groupby(["model", "item_key"]).cumcount() + 1
results = results[["item_key", "model", "config", "trial", "outcome"]].astype({"trial": "int64", "outcome": "float64"})
meta = pd.concat([results[["item_key", "model", "config", "trial"]], pd.DataFrame(meta)], axis=1)
assert not results.duplicated(["item_key", "model", "trial"]).any()
assert set(results.item_key) <= set(items.item_key)

os.makedirs(OUT, exist_ok=True)
items.to_parquet(f"{OUT}/items.parquet", index=False)
results.to_parquet(f"{OUT}/results.parquet", index=False)
meta.to_parquet(f"{OUT}/results_meta.parquet", index=False)

# ---------------- summary
base = results.model.str.replace(r" \(.*$", "", regex=True)
yr = base.map(RELEASE).str[:4]
print("items", len(items), "| items with results", results.item_key.nunique())
print("results", len(results), "| dropped", len(dropped))
for d in dropped:
    print("  dropped:", d)
print("models", results.model.nunique(), "| models from 2025-2026",
      results.assign(y=yr).query("y in ['2025','2026']").model.nunique())
print("outcome values", results.outcome.round(3).value_counts().sort_index().to_dict())
print("non-binary outcomes", int((~results.outcome.isin([0.0, 1.0])).sum()))
print("n_grades", meta.n_grades.value_counts().sort_index().to_dict(),
      "| repeated judge rows", int((meta.n_distinct_judges < meta.n_grades).sum()),
      "| auto-graded", int(meta.auto_graded.sum()), "| any_uncertain", int(meta.any_uncertain.sum()))
print(results.assign(split=meta.hf_split).groupby(["split", "model"]).agg(
    n=("outcome", "size"), items=("item_key", "nunique"), max_trial=("trial", "max"), mean=("outcome", "mean")).round(3).to_string())
print("items per hf split:", {s: int(items.extra.apply(lambda e: s in json.loads(e)["hf_splits"]).sum()) for s in SPLITS})
print("items sharing text with another item:", [k for k, e in zip(items.item_key, items.extra) if json.loads(e)["same_text_as"]])
print("text length chars", items.text.str.len().describe().round(0).to_dict())
