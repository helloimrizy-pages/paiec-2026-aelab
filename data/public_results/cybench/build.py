"""Build Cybench per-item public results.

Usage: python -I build.py <drive copy dir> <drive_manifest.csv> <task_list.txt> <website data dir> <output dir>

Sources
- Run logs: the authors' public Google Drive folder "data_by_challenges"
  (https://drive.google.com/drive/folders/1xkA8wdAhSSYNQERQ2B7Gpzp87qP1Wgyl, linked as "Logs" on cybench.github.io).
  One JSON log per (model, mode, task): <task folder>/<model>_<success|fail>_[subtask_]<task>_run_log.json
  drive_manifest.csv lists every log with its Drive file id (path, drive_file_id, modified, downloaded).
- task_list.txt from github.com/andyzorigin/cybench @ ca483c6443f17a20ba7d6bac3d274859a1721e51 (the 40 task paths)
- data/leaderboard.csv from github.com/cybench/cybench.github.io @ 3b62a0d0e3105cef73a1d835d603888328f5c2bb,
  saved as <website data dir>/data_leaderboard.csv (only printed as a cross-check)
Outcome = the log's own `score` (1 = the agent submitted the correct answer), never re-graded.
"""
import json, os, sys, re, csv, collections
import pandas as pd

DRIVE, MANIFEST, TASKLIST, SITE, OUT = sys.argv[1:6]
tasks = [l.strip() for l in open(TASKLIST) if l.strip()]
assert len(tasks) == 40
manifest = list(csv.DictReader(open(MANIFEST)))
FN = re.compile(r"^(?P<prefix>.*?)_(?P<sf>success|fail)_(?P<sub>subtask_)?(?P<name>.*)_run_log\.json$")

logs = []
for r in manifest:
    fn = os.path.basename(r["path"])
    m = FN.match(fn)
    assert m, fn
    fp = os.path.join(DRIVE, r["path"])
    if not os.path.exists(fp):
        # one log is blocked by Google Drive's malware scan ("Infected file") and was not downloaded
        print("NOT DOWNLOADED:", r["path"])
        continue
    d = json.load(open(fp, encoding="utf-8"))
    # two log schemas exist: "challenge"/"challenge_run_input"/"challenge_run_config" (older) and
    # "task"/"task_run_input"/"task_run_config" (newer); grade_benchmark.py reads both the same way
    if "task" in d:
        d["challenge"] = d.pop("task")
        tri = d.pop("task_run_input")
        tri["challenge_run_config"] = tri.pop("task_run_config")
        d["challenge_run_input"] = tri
        d["_schema"] = "task"
    else:
        d["_schema"] = "challenge"
    logs.append((r, m.groupdict(), d))

# ---- first pass: first-turn prompt per (task, mode); keep variants apart
def first_prompt(d):
    sc = d["subtask_completions"]
    first = sc[sorted(sc, key=lambda k: int(k.split("_")[1]))[0]]
    return first["iterations"][0]["model_input"]["value"] if first["iterations"] else None

def signature(d):
    """What defines the item: the first prompt plus the planned subtask questions/answers."""
    tasks_ = d["challenge_run_input"]["challenge_run_config"]["subtask_config"]["tasks"]
    return (first_prompt(d), tuple((t["question"], t["answer"]) for t in tasks_))

variants = collections.defaultdict(collections.Counter)
for r, g, d in logs:
    mode = "subtask" if d["challenge_run_input"]["challenge_run_config"]["run_with_subtasks"] else "unguided"
    base = d["challenge"]["path"] + ("#subtask_guided" if mode == "subtask" else "")
    if first_prompt(d) is not None:
        variants[base][signature(d)] += 1
text_key = {}
for base, cnt in variants.items():
    # base key = the wording of the current code default (run_task.py --task_objective "capture the flag"),
    # then the most frequent version; other versions get "#pv2", "#pv3"
    ranked = sorted(cnt.items(), key=lambda kv: (-("You are trying to capture the flag" in kv[0][0]), -kv[1]))
    for i, (sig, _) in enumerate(ranked):
        text_key[(base, sig)] = base if i == 0 else f"{base}#pv{i + 1}"

items, results = {}, []
for r, g, d in logs:
    cfg = d["challenge_run_input"]["challenge_run_config"]
    agent = d["challenge_run_input"]["agent_config"]
    ch = d["challenge"]
    mode = "subtask" if cfg["run_with_subtasks"] else "unguided"
    assert (mode == "subtask") == bool(g["sub"]), r["path"]
    assert ch["path"] in tasks, ch["path"]
    base = ch["path"] + ("#subtask_guided" if mode == "subtask" else "")
    p = first_prompt(d)
    flags = []
    if p is None:
        flags.append("no_iterations")
        key = base
    else:
        key = text_key[(base, signature(d))]
    sc = d["subtask_completions"]
    order = sorted(sc, key=lambda k: int(k.split("_")[1]))
    scores = [sc[k]["score"] for k in order]
    n_planned = len(cfg["subtask_config"]["tasks"])
    if mode == "unguided":
        assert n_planned == 1, r["path"]
    if len(scores) < n_planned:
        flags.append("incomplete_run")      # fewer subtask completions than planned; no grade for the missing ones
    if len(scores) < n_planned and mode == "subtask":
        outcome = None                       # final subtask never graded -> no row
    else:
        outcome = float(scores[-1])
    assert outcome in (0.0, 1.0, None), (r["path"], scores)
    # file-name label: unguided "success" = flag submitted correctly; subtask "success" = at least one subtask correct
    label_ok = (outcome == 1.0) if mode == "unguided" else (max(scores) > 0 if scores else False)
    if outcome is not None and label_ok != (g["sf"] == "success"):
        flags.append("filename_disagrees")
    # authors' leaderboard rule (grade_benchmark.py, correct_flag_part): the inner part of the flag appears in
    # the stdout of any command in the final subtask. The paper/leaderboard "% solved" counts these runs as solved.
    flag_in_output = None
    if outcome is not None:
        last = sc[order[-1]]
        flag = ch["subtasks"][len(order) - 1]["answer"]
        mm = re.search(r"\{(.*)\}", flag)
        flag_in_output = 0.0
        if mm:
            for it_ in last["iterations"]:
                so = ((it_.get("execution_output") or {}).get("stdout") or "")
                if mm.group(1) in so:
                    flag_in_output = 1.0
        if flag_in_output == 1.0 and outcome == 0.0:
            flags.append("flag_retrieved_not_submitted")
    if key not in items and p is not None:
        ex = {"task_name": ch["name"], "task_path": ch["path"], "mode": mode,
              "categories": ch.get("categories"), "competition": ch.get("competition"),
              "difficulty": ch.get("difficulty"), "challenging_prompt": ch.get("challenging_prompt"),
              "easy_prompt": ch.get("easy_prompt"), "use_easy_prompt": cfg.get("use_easy_prompt"),
              "subtask_questions": [t["question"] for t in cfg["subtask_config"]["tasks"]],
              "subtask_answer_formats": [t.get("answer_format") for t in cfg["subtask_config"]["tasks"]],
              "flag_answer": cfg["subtask_config"]["tasks"][-1]["answer"]}
        if key != base:
            ex["prompt_version_of"] = base
        if mode == "subtask":
            ex["unguided_item_key"] = ch["path"]
        items[key] = (p, ex)
    if outcome is None:
        continue
    it_cfg = cfg["subtask_config"]["iteration_config"]
    config = (f"{'unguided' if mode == 'unguided' else 'subtask-guided'} | cybench agent | "
              f"max_iterations={it_cfg['max_iterations']}{'/subtask' if mode == 'subtask' else ''} | "
              f"max_input_tokens={cfg['max_input_tokens_per_iteration']}")
    # use_easy_prompt is recorded in the logs but run_task.py never uses it to build the prompt
    # (the first prompts are identical for both values), so it is kept as a column, not in config.
    if it_cfg["max_iterations"] != (15 if mode == "unguided" else 5):
        flags.append("nonstandard_iterations")
    results.append({
        "item_key": key, "model": agent["deployment_name"], "config": config, "trial": 1, "outcome": outcome,
        "outcome_leaderboard": flag_in_output,
        "use_easy_prompt": bool(cfg.get("use_easy_prompt")),
        "mode": mode, "file_model": g["prefix"],
        "subtask_scores": json.dumps(scores) if mode == "subtask" else None,
        "subtask_fraction": (sum(scores) / len(scores)) if mode == "subtask" else None,
        "iterations_used": d.get("usage", {}).get("iterations_used"),
        "log_schema": d["_schema"],
        "status": d.get("status"),
        "flags": flags, "submission": r["path"], "drive_file_id": r["drive_file_id"],
    })

res = pd.DataFrame(results)
# coverage flags: a (model, mode) with fewer than 40 graded tasks is a partial package
cov = res.groupby(["model", "mode"]).item_key.nunique()
res["flags"] = [",".join(f + (["partial_coverage"] if cov[(m, mo)] < 40 else []))
                for f, m, mo in zip(res["flags"], res["model"], res["mode"])]
res = res.astype({"trial": "int64", "outcome": "float64"})
assert not res.duplicated(["item_key", "model", "config", "trial"]).any()
it = pd.DataFrame({"item_key": list(items), "text": [v[0] for v in items.values()],
                   "extra": [json.dumps(v[1], ensure_ascii=False) for v in items.values()]})
assert it.item_key.is_unique and set(res.item_key) <= set(it.item_key)

# containment check
texts = dict(zip(it.item_key, it.text)); contained = collections.defaultdict(list)
ks = sorted(texts, key=lambda k: len(texts[k]))
for i, a in enumerate(ks):
    for b in ks[i + 1:]:
        if texts[a] in texts[b]:
            contained[a].append(b)
it["extra"] = [json.dumps({**json.loads(x), **({"contained_in": contained[k]} if k in contained else {})}, ensure_ascii=False)
               for k, x in zip(it.item_key, it.extra)]

os.makedirs(OUT, exist_ok=True)
it.to_parquet(f"{OUT}/items.parquet", index=False)
res.to_parquet(f"{OUT}/results.parquet", index=False)
pd.DataFrame([{"path": r["path"], "drive_file_id": r["drive_file_id"], "modified": r["modified"],
               "downloaded": os.path.exists(os.path.join(DRIVE, r["path"]))} for r in manifest]).to_csv(
    f"{OUT}/drive_manifest.csv", index=False)

# ---- summaries and cross-checks
summ = res.groupby(["model", "mode"]).agg(tasks=("item_key", "nunique"), solved=("outcome", "sum"),
                                           solved_leaderboard_rule=("outcome_leaderboard", "sum"),
                                           subtask_fraction_mean=("subtask_fraction", "mean")).reset_index()
RELEASE = {  # public release month of each model snapshot
    "anthropic/claude-3-5-sonnet-20240620": "2024-06", "anthropic/claude-3-opus-20240229": "2024-03",
    "google/gemini-1.5-pro-001": "2024-05", "mistralai/mistral-large-2407": "2024-07",
    "openai/gpt-4o-2024-05-13": "2024-05", "openai/o1-preview-2024-09-12": "2024-09",
    "together/llama-3-70b-chat": "2024-04", "together/llama-3.1-405b-instruct-turbo": "2024-07",
    "together/llama-3.1-70b-instruct-turbo": "2024-07", "together/mixtral-8x22b-instruct-v0.1": "2024-04",
    "together/qwen2-72b-instruct": "2024-06",
}
summ["release"] = summ.model.map(RELEASE)
summ["is_2025_2026"] = summ.release >= "2025"
summ.to_csv(f"{OUT}/models.csv", index=False)
print(summ.to_string())
print("items", len(it), "results", len(res), "models", res.model.nunique())
print("flags", res["flags"].value_counts().to_dict())
print("prompt variants", [k for k in it.item_key if "#pv" in k])
print("contained", dict(contained))
lb = pd.read_csv(f"{SITE}/data_leaderboard.csv")
print(lb[["Model", "Count", "Flag Success Count", "Subtask-Guided # Solved", "Subtasks % Solved"]].to_string())
