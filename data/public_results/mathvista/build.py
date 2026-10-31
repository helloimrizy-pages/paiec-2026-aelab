"""Build items.parquet / results.parquet / runs.csv for MathVista testmini.

Usage: python -I build.py <download_dir> <output_dir>   (fetch.py downloads the sources and runs this)

<download_dir> layout:
  MathVista/   checkout of https://github.com/lupantech/MathVista at c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb
               (results/<model>/output_*.json = per-problem outputs with a `true_false` grade)
  hf/data/testmini-00000-of-00001-725687bf7a18d64b.parquet
               from HF dataset AI4Math/MathVista (https://huggingface.co/datasets/AI4Math/MathVista)
               at revision 2b6ad69445fbb5695c9b165475e8decdbeb97747
Outcome = the run file's `true_false` grade (true -> 1.0, false -> 0.0); a null grade gets no row.
"""
import glob, json, os, sys
import pandas as pd

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
R = f"{D}/MathVista/results"

# run file -> (model display name, API/model id, config, release month)
LLM = {"chatgpt": ("ChatGPT", "gpt-3.5-turbo", "2023-03"),
       "gpt4": ("GPT-4", "gpt-4-0613", "2023-06"),
       "claude": ("Claude-2", "claude-2", "2023-07")}
SETTINGS = {"": "zero-shot; text-only, query only (no image)",
            "_2shot_solution": "2-shot CoT; text-only, query only (no image)",
            "_2shot_solution_use_caption": "2-shot CoT; text-only, query + Bard image caption",
            "_2shot_solution_use_ocr": "2-shot CoT; text-only, query + EasyOCR image text",
            "_2shot_solution_use_caption_ocr": "2-shot CoT; text-only, query + Bard image caption + EasyOCR image text",
            "_2shot_code": "2-shot PoT; text-only, query only (no image)",
            "_2shot_code_use_caption": "2-shot PoT; text-only, query + Bard image caption",
            "_2shot_code_use_ocr": "2-shot PoT; text-only, query + EasyOCR image text",
            "_2shot_code_use_caption_ocr": "2-shot PoT; text-only, query + Bard image caption + EasyOCR image text"}
RUNS = {}
for d, (name, api, rel) in LLM.items():
    for suf, cfg in SETTINGS.items():
        f = f"{d}/output_{d}{suf}.json"
        if os.path.exists(f"{R}/{f}"):
            RUNS[f] = (name, api, cfg, rel)
LMM_CFG = "zero-shot; image + query (multimodal)"
RUNS.update({
    "bard/output_bard.json": ("Multimodal Bard", "Google Bard chatbot (bardapi)", LMM_CFG, "2023-07"),
    "idefics/output_idefics_9b_instruct.json": ("IDEFICS (9B-Instruct)", "HuggingFaceM4/idefics-9b-instruct", LMM_CFG, "2023-08"),
    "instruct_blip2/output_instruct_blip2_vicuna_13b.json": ("InstructBLIP (Vicuna-7B)", "InstructBLIP (file name says vicuna_13b; leaderboard says Vicuna-7B)", LMM_CFG, "2023-05"),
    "llama_adapter_v2/output_llama_adapter_v2.json": ("LLaMA-Adapter-V2 (7B)", "LLaMA-Adapter V2 multimodal 7B", LMM_CFG, "2023-04"),
    "llava/output_llava_llama_2_13b.json": ("LLaVA (LLaMA-2-13B)", "LLaVA LLaMA-2-13B-chat", LMM_CFG, "2023-07"),
    "llavar/output_llavar.json": ("LLaVAR", "LLaVAR", LMM_CFG, "2023-06"),
    "minigpt4/output_minigpt4_llama2.json": ("miniGPT4 (LLaMA-2-7B)", "MiniGPT-4 LLaMA-2-7B-chat", LMM_CFG, "2023"),
    "mplugowl/output_mplugowl_7b_ft.json": ("mPLUG-Owl (LLaMA-7B)", "mPLUG-Owl 7B (ft)", LMM_CFG, "2023-04"),
})
# every per-problem output file in results/ must be accounted for (only non-model baselines/templates are skipped)
SKIP = {"random/output_frequent_guess.json", "random/output_random_guess.json",
        "test_random/output_test_frequent_guess.json", "test_random/output_test_random_guess.json",
        "leaderboad_submission_template/output_test_template_for_leaderboard_submission.json",
        "leaderboad_submission_template/output_testmini_template_for_leaderboard_submission.json"}
found = {os.path.relpath(p, R) for p in glob.glob(f"{R}/*/output_*.json")}
assert found == set(RUNS) | SKIP, found ^ (set(RUNS) | SKIP)

# ------------------------------------------------------------------ items (HF testmini)
hf = pd.read_parquet(glob.glob(f"{D}/hf/data/testmini-*.parquet")[0],
                     columns=["pid", "question", "image", "choices", "unit", "precision", "answer",
                              "question_type", "answer_type", "metadata", "query"])
assert len(hf) == 1000 and hf.pid.is_unique
cap_run = json.load(open(f"{R}/gpt4/output_gpt4_2shot_solution_use_caption_ocr.json"))
old_query = json.load(open(f"{R}/bard/output_bard.json"))  # the query string the 2023 runs used
dup = hf.groupby("query").pid.transform("size")


def clean(v):
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, float) and v != v:
        return None
    return v


rows = []
for (_, r), n in zip(hf.iterrows(), dup):
    pid = r.pid
    ex = {k: clean(r[k]) for k in ["question", "choices", "unit", "precision", "answer", "question_type",
                                    "answer_type", "image", "metadata"]}
    ex["text_source"] = "HF AI4Math/MathVista testmini `query` (hint + question + choices)"
    ex["n_items_sharing_text"] = int(n)
    ex["bard_caption"] = cap_run[pid].get("caption") or None
    ex["easyocr_text"] = cap_run[pid].get("ocr") or None
    if old_query[pid]["query"] != r["query"]:
        ex["query_in_2023_outputs"] = old_query[pid]["query"]
    rows.append({"item_key": pid, "text": r["query"], "extra": json.dumps(ex, ensure_ascii=False)})
items = pd.DataFrame(rows)
assert items.item_key.is_unique
items.to_parquet(f"{OUT}/items.parquet", index=False)

# ------------------------------------------------------------------ results
gt = dict(zip(hf.pid, hf.answer))
res, runs = [], []
for f, (name, api, cfg, rel) in sorted(RUNS.items()):
    o = json.load(open(f"{R}/{f}"))
    assert set(o) == set(gt), f
    n_ok = n_none = n_true = 0
    for pid, v in o.items():
        assert str(v["answer"]) == gt[pid], (f, pid)
        tf = v.get("true_false")
        if tf is True or tf is False:
            res.append({"item_key": pid, "model": name, "config": cfg, "trial": 1, "outcome": float(tf)})
            n_ok += 1; n_true += tf
        else:
            n_none += 1  # no explicit grade -> no row
    sc = json.load(open(f"{R}/{os.path.dirname(f)}/" + os.path.basename(f).replace("output_", "scores_", 1)))
    assert sc["average"]["correct"] == n_true and sc["average"]["total"] == 1000, (f, sc["average"], n_true)
    runs.append({"run_file": f"results/{f}", "model": name, "model_id": api, "config": cfg, "release": rel,
                 "n_graded": n_ok, "n_ungraded": n_none, "n_correct": n_true,
                 "official_accuracy": sc["average"]["accuracy"]})
results = pd.DataFrame(res).astype({"trial": "int64", "outcome": "float64"})
assert not results.duplicated(["item_key", "model", "config", "trial"]).any()
results.to_parquet(f"{OUT}/results.parquet", index=False)
pd.DataFrame(runs).to_csv(f"{OUT}/runs.csv", index=False)

print(pd.DataFrame(runs)[["run_file", "n_graded", "n_ungraded", "n_correct", "official_accuracy"]].to_string())
print("items", len(items), "distinct text", items.text.nunique(),
      "items whose text is shared", int((dup > 1).sum()), "dup groups", int(hf[dup > 1]["query"].nunique()),
      "distinct question", hf.question.nunique())
print("results", len(results), "models", results.model.nunique(), "runs", results.groupby(["model", "config"]).ngroups)
print("items with query changed since 2023 runs", sum("query_in_2023_outputs" in json.loads(e) for e in items.extra))
print("items with bard caption", sum(json.loads(e)["bard_caption"] is not None for e in items.extra))
