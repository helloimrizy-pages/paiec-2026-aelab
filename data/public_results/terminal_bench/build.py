"""Build normalized per-item public results for Terminal-Bench (versions 2.0, 2.1, 3.0, 4.0).

Usage: python -I build.py <download_dir> <output_dir>

<download_dir> layout (produced by fetch.py):
  hub/boards.json          Harbor Hub leaderboards 2.0/2.1/3.0/4.0: rows, row->trial ids, dataset versions + task lists
  hub/trials.json          Hub trial records for every row trial (task_name, task_content_hash, rewards, exception_type, config)
  hub/task_versions.json   Hub registry task_version rows (instruction text) for every content hash seen
                           (Hub = https://ofhuhcpkvzjlejydnvyd.supabase.co, public read API, not versioned; see fetch_hub.py)
  raw_tb21/submissions/    github.com/harbor-framework/terminal-bench-2-1 @7131e4375048a0e408a8fb404b5f499d726b695b
                           leaderboard/submissions/*.json (disqualified_trials)
  raw_lb/submissions/      github.com/harbor-framework/terminal-bench @2e29ca91d5f1cb6fb60924bf867acc755565c044
                           leaderboard/submissions/*.json (TB 4.0; disqualified_trials)
  hf_tb2_git/submissions/terminal-bench/2.0/<submission>/{metadata.yaml,<job>/<trial>/result.json}
                           sparse git checkout of https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard
                           @572b2614be2c0cb2527e14f5b1e4026f1072e6c1
  git/export/<ref>/...     instruction.md files at the GIT_REFS below (cross-check only)
Outcome (leaderboard/src/leaderboard/core/metrics.py, is_success): 1.0 iff reward > 0, else 0.0. Trials with no
reward (errored) get no results row. TB 2.1 reward-hacking disqualifications: when every success of a (run, task)
pair was disqualified those trials are 0.0; when only some were, all trials of the pair are dropped.
"""
import collections, glob, json, os, sys
import pandas as pd
import yaml

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
EXP = f"{D}/git/export"
GIT_REFS = {"2.0": ("tb2_head", "harbor-framework/terminal-bench-2@2fd12b88aafdd04a52c298e3940bcb189f9766d6"),
            "2.1": ("tb21_head", "harbor-framework/terminal-bench-2-1@7131e4375048a0e408a8fb404b5f499d726b695b"),
            "3.0": ("tb_v3.0.0", "harbor-framework/terminal-bench@v3.0.0"),
            "4.0": ("tb_v4.0.0", "harbor-framework/terminal-bench@v4.0.0")}


# Hub registry package names cannot contain "."; the git folder / HF trial name keeps it.
HUB_NAME = {"install-windows-3.11": "install-windows-3-11"}
GIT_NAME = {v: k for k, v in HUB_NAME.items()}


def git_instruction(ver, task):
    sub, _ = GIT_REFS[ver]
    task = GIT_NAME.get(task, task)
    for p in (f"{EXP}/{sub}/tasks/{task}/instruction.md", f"{EXP}/{sub}/{task}/instruction.md"):
        if os.path.exists(p):
            return open(p, encoding="utf-8").read()
    return None


def is_success(r):  # leaderboard/src/leaderboard/core/metrics.py: success iff reward > 0
    return r is not None and float(r) > 0


# ------------------------------------------------------------------ Hub registry: task texts
snap = json.load(open(f"{D}/hub/boards.json"))
trials = {t["id"]: t for t in json.load(open(f"{D}/hub/trials.json"))}
tv_by_hash = collections.defaultdict(list)
for tv in json.load(open(f"{D}/hub/task_versions.json")):
    tv_by_hash[tv["content_hash"]].append(tv)


def tv_for(h, task_name):
    """task_version row for a content hash, restricted to the package named by the task (one hash is
    shared by an unrelated third-party package)."""
    org, name = task_name.split("/", 1)
    c = [tv for tv in tv_by_hash.get(h, []) if tv["package"]["name"] == name and tv["package"]["org"]["name"] == org]
    if not c:  # trial ran a re-packaged copy (TB 4.0 "ryanmarten/terminal-bench-4-no-prebuild-<task>")
        c = [tv for tv in tv_by_hash.get(h, []) if tv["package"]["name"].endswith("-" + name)]
    assert len(c) == 1, (h, task_name, len(c))
    return c[0]


items = {}  # item_key -> dict
for ver, b in snap["boards"].items():
    hashes = collections.defaultdict(set)
    for links in b["dataset_version_tasks"].values():
        for l in links:
            tv = l["task_version"]
            hashes[f"{tv['package']['org']['name']}/{tv['package']['name']}"].add(tv["content_hash"])
    for r in b["rows"]:
        for i in r["trial_ids"]:
            hashes[trials[i]["task_name"]].add(trials[i]["task_content_hash"])
    for task_name, hs in sorted(hashes.items()):
        tvl = [tv_for(h, task_name) for h in sorted(hs)]
        texts = {tv["instruction"] for tv in tvl}
        assert len(texts) == 1, (ver, task_name, len(texts))
        tv = max(tvl, key=lambda x: x["published_at"])
        task = task_name.split("/", 1)[1]
        g = git_instruction(ver, task)
        md = tv.get("metadata") or {}
        items[f"{ver}/{task}"] = {
            "item_key": f"{ver}/{task}", "text": tv["instruction"],
            "extra": {"version": ver, "task_id": task, "task_name": task_name,
                      "text_source": "harbor_hub_registry.task_version.instruction",
                      "content_hashes": sorted(hs), "description": tv.get("description"),
                      "category": md.get("category"), "subcategory": md.get("subcategory"), "tags": md.get("tags"),
                      "difficulty": md.get("difficulty"), "expert_time_estimate_hours": md.get("expert_time_estimate_hours"),
                      "junior_time_estimate_hours": md.get("junior_time_estimate_hours"),
                      "author_name": md.get("author_name"), "published_at": tv.get("published_at"),
                      "git_ref": GIT_REFS[ver][1],
                      "git_instruction_identical": None if g is None else (g == tv["instruction"]),
                      "git_instruction_identical_stripped": None if g is None else (g.strip() == tv["instruction"].strip())}}

# ------------------------------------------------------------------ Hub results (2.1, 3.0, 4.0)
res, runs, details = [], [], []


def label(x):
    return x.get("label") if isinstance(x, dict) else x


subs = {"2.1": [json.load(open(p)) | {"_file": os.path.basename(p)} for p in sorted(glob.glob(f"{D}/raw_tb21/submissions/*.json"))],
        "4.0": [json.load(open(p)) | {"_file": os.path.basename(p)} for p in sorted(glob.glob(f"{D}/raw_lb/submissions/*.json"))]}

for ver in ("2.1", "3.0", "4.0"):
    for row in snap["boards"][ver]["rows"]:
        ts = [trials[i] for i in row["trial_ids"]]
        a0 = ts[0]["config"]["agent"]
        model = a0["model_name"]
        agent = a0["name"]
        assert all(t["config"]["agent"]["model_name"] == model and t["config"]["agent"]["name"] == agent for t in ts)
        efforts = {(t["config"]["agent"].get("kwargs") or {}).get("reasoning_effort") for t in ts}
        assert len(efforts) == 1, efforts
        effort = efforts.pop()
        avers = sorted({(t["config"]["agent"].get("kwargs") or {}).get("version") for t in ts} - {None})
        config = " ".join([agent] + avers + ([f"effort={effort}"] if effort else [])) + f" | {ver}"
        md = row["metadata"]
        canon = {l["task_version"]["content_hash"] for links in snap["boards"][ver]["dataset_version_tasks"].values() for l in links}
        pkgs = collections.Counter(tv_for(t["task_content_hash"], t["task_name"])["package"]["org"]["name"]
                                   for t in ts if t["task_content_hash"] not in canon)
        raw_succ = sum(is_success((t["rewards"] or {}).get("reward")) for t in ts)
        board_acc = row["metrics"]["accuracy"]
        exp_dq = raw_succ - round(board_acc * len(ts) / 100)
        # ---- disqualifications (reward hacking) from the repo submission that matches this row
        dq_by_task, dq_file = collections.Counter(), None
        if ver in subs:
            cands = [s for s in subs[ver]
                     if s["source_filter"]["agent"] == agent
                     and s["source_filter"]["model_name"].split("/")[-1] == model.split("/")[-1]
                     and s["source_filter"].get("reasoning_effort") == effort
                     and abs(s["metrics"]["accuracy"] - board_acc) < 0.006]
            if len(cands) == 1:
                dq_file = cands[0]["_file"]
                for d in cands[0].get("disqualified_trials") or []:
                    dq_by_task["terminal-bench/" + d["judge_trial"].rstrip("/").split("/")[-1]] += 1
            assert exp_dq <= 0 or dq_file, (ver, model, config)
        by_task = collections.defaultdict(list)
        for t in ts:
            by_task[t["task_name"]].append(t)
        n_rows = n_dq0 = n_amb = n_null = 0
        for task_name, tl in by_task.items():
            tl.sort(key=lambda t: (t.get("started_at") or "", t["trial_name"]))
            succ_ids = [t["id"] for t in tl if is_success((t["rewards"] or {}).get("reward"))]
            ndq = dq_by_task.get(task_name, 0)
            for k, t in enumerate(tl, 1):
                rew = (t["rewards"] or {}).get("reward")
                key = f"{ver}/{task_name.split('/', 1)[1]}"
                assert key in items
                note, outcome = "", None
                if rew is None:
                    note = "no_reward_errored"; n_null += 1
                elif ndq and ndq >= len(succ_ids) and t["id"] in succ_ids:
                    # every success of this (run, task) was disqualified -> the board's explicit grade is 0
                    note, outcome = "disqualified_reward_hacking", 0.0; n_dq0 += 1
                elif ndq and ndq >= len(succ_ids):
                    outcome = 1.0 if is_success(rew) else 0.0  # a failed trial in a fully-identified pair
                elif ndq:  # some, but not all, successes disqualified and the trial ids are gone: cannot tell which
                    note = "dropped_ambiguous_disqualification"; n_amb += 1
                else:
                    outcome = 1.0 if is_success(rew) else 0.0
                if outcome is not None:
                    res.append({"item_key": key, "model": model, "config": config, "trial": k, "outcome": outcome})
                    n_rows += 1
                details.append({"item_key": key, "model": model, "config": config, "trial": k, "outcome": outcome,
                                "reward_raw": None if rew is None else float(rew), "exception_type": t.get("exception_type"),
                                "note": note, "trial_id": t["id"], "run_id": f"hub:{row['id']}"})
        runs.append({"run_id": f"hub:{row['id']}", "version": ver, "source": "harbor_hub_leaderboard",
                     "model": model, "config": config, "model_display": label(md.get("model_display")),
                     "agent_display": label(md.get("agent_display")), "model_org": label(md.get("model_org")),
                     "board_date_field": md.get("date"), "release_date_field": md.get("release_date"),
                     "n_trials": len(ts), "n_tasks": len(by_task), "n_results": n_rows,
                     "n_success_raw": raw_succ, "board_accuracy": board_acc,
                     "recomputed_accuracy_raw": round(100 * raw_succ / len(ts), 2), "board_expected_dq": exp_dq,
                     "dq_listed": sum(dq_by_task.values()), "dq_source_file": dq_file,
                     "n_set_to_0_dq": n_dq0, "n_dropped_ambiguous_dq": n_amb, "n_no_reward": n_null,
                     "n_trials_noncanonical_task_hash": sum(pkgs.values()),
                     "noncanonical_task_package_orgs": json.dumps(dict(pkgs)) if pkgs else None,
                     "job_ids": ",".join(sorted({t["job_id"] for t in ts}))})

# ------------------------------------------------------------------ TB 2.0 (HF terminal-bench-2-leaderboard)
TB2 = f"{D}/hf_tb2_git/submissions/terminal-bench/2.0"
# metadata.yaml errors fixed here (each contradicts both the folder name and the trial-level model name), and
# folders without metadata.yaml (model taken from the folder name).
MODEL_OVERRIDE = {"Ante__Gemini-3.1-Pro-Preview": "gemini-3.1-pro-preview",            # metadata says gemini-3-pro-preview
                  "LemonCode__GPT-5.3-Codex": "gpt-5.3-codex + gemini-3.1-pro-preview",  # metadata says gpt-3.5-codex
                  "logos-latest__claude-opus-4.7": "claude-opus-4.7",                  # no metadata.yaml
                  "logos-ts__claude-opus-4.7": "claude-opus-4.7"}                      # no metadata.yaml
for folder in sorted(d for d in os.listdir(TB2) if os.path.isdir(f"{TB2}/{d}")):
    meta_p = f"{TB2}/{folder}/metadata.yaml"
    meta = yaml.safe_load(open(meta_p)) if os.path.exists(meta_p) else {}
    mnames = [str(m.get("model_name")) for m in meta.get("models") or []]
    model = MODEL_OVERRIDE.get(folder) or " + ".join(mnames)
    disp = meta.get("agent_display_name") or folder.split("__")[0]
    tl = []
    for job in sorted(d for d in os.listdir(f"{TB2}/{folder}") if os.path.isdir(f"{TB2}/{folder}/{d}")):
        for tdir in sorted(os.listdir(f"{TB2}/{folder}/{job}")):
            p = f"{TB2}/{folder}/{job}/{tdir}/result.json"
            if os.path.exists(p):
                r = json.load(open(p))
                assert "task_name" in r and "verifier_result" in r, p
                tl.append((job, r))
    agents = collections.Counter((r.get("agent_info") or {}).get("name") for _, r in tl)
    main_agent = agents.most_common(1)[0][0]
    avers = collections.Counter((r.get("agent_info") or {}).get("version") for _, r in tl
                                if (r.get("agent_info") or {}).get("name") == main_agent)
    av = [a for a in avers if a and a != "unknown"]
    efforts = {((r.get("config") or {}).get("agent", {}).get("kwargs") or {}).get("reasoning_effort") for _, r in tl} - {None}
    config = " ".join([str(disp)] + ([str(av[0])] if len(av) == 1 else []) +
                      ([f"effort={next(iter(efforts))}"] if len(efforts) == 1 else [])) + " | 2.0"
    trial_models = collections.Counter((r.get("config") or {}).get("agent", {}).get("model_name") for _, r in tl)
    by_task = collections.defaultdict(list)
    for job, r in tl:
        t = r["task_name"].split("/")[-1]  # some submissions record "terminal-bench/<task>"
        by_task[HUB_NAME.get(t, t)].append((job, r))
    n_rows = n_null = raw_succ = n_other_agent = 0
    for task, lst in by_task.items():
        lst.sort(key=lambda x: (x[0], x[1].get("started_at") or "", x[1].get("trial_name") or ""))
        key = f"2.0/{task}"
        assert key in items, key
        k = 0
        for job, r in lst:
            vr = r.get("verifier_result") or {}
            rew = (vr.get("rewards") or {}).get("reward")
            exc = (r.get("exception_info") or {}).get("exception_type")
            outcome, note, trial_k = None, "", None
            if (r.get("agent_info") or {}).get("name") != main_agent:
                note = "dropped_other_agent_in_submission"; n_other_agent += 1
            elif rew is None:
                note = "no_reward_errored"; n_null += 1
            else:
                outcome = 1.0 if is_success(rew) else 0.0
            if note != "dropped_other_agent_in_submission":
                k += 1; trial_k = k
            if outcome is not None:
                raw_succ += outcome == 1.0
                res.append({"item_key": key, "model": model, "config": config, "trial": trial_k, "outcome": outcome})
                n_rows += 1
            details.append({"item_key": key, "model": model, "config": config, "trial": trial_k, "outcome": outcome,
                            "reward_raw": None if rew is None else float(rew), "exception_type": exc, "note": note,
                            "trial_id": r.get("id"), "run_id": f"hf2:{folder}"})
    n_kept = len(tl) - n_other_agent
    runs.append({"run_id": f"hf2:{folder}", "version": "2.0", "source": "hf_terminal-bench-2-leaderboard",
                 "model": model, "config": config,
                 "model_display": " + ".join(str(m.get("model_display_name")) for m in meta.get("models") or []) or None,
                 "agent_display": disp,
                 "model_org": " + ".join(str(m.get("model_org_display_name")) for m in meta.get("models") or []) or None,
                 "metadata_model_names": " + ".join(mnames) or None, "model_overridden": folder in MODEL_OVERRIDE,
                 "trial_model_names": json.dumps(dict(trial_models)), "agent_names": json.dumps(dict(agents)),
                 "n_jobs": len({j for j, _ in tl}), "n_trials": len(tl), "n_tasks": len(by_task),
                 "trials_per_task": json.dumps(sorted({len(v) for v in by_task.values()})), "n_results": n_rows,
                 "n_success_raw": raw_succ, "recomputed_accuracy_raw": round(100 * raw_succ / n_kept, 2),
                 "n_no_reward": n_null, "n_dropped_other_agent": n_other_agent})

# ------------------------------------------------------------------ model release dates
# Hub boards: 2.0/2.1/4.0 rows carry the model release date in metadata.date (one value per model, matching
# public release dates where known); 3.0 rows carry metadata.release_date (their metadata.date is the run date).
def norm(m):
    return m.lower().split("/")[-1].replace(".", "-").replace("_", "-")
rel = {}
for r in snap["boards"]["2.0"]["rows"]:
    if len(r["metadata"]["model_names"]) == 1:
        rel.setdefault(norm(r["metadata"]["model_names"][0]), (r["metadata"]["date"], "hub_tb2.0_board_date"))
for run in runs:
    if run["source"] == "harbor_hub_leaderboard":
        d = run["release_date_field"] if run["version"] == "3.0" else run["board_date_field"]
        rel.setdefault(norm(run["model"]), (d, f"hub_tb{run['version']}_board_{'release_date' if run['version'] == '3.0' else 'date'}"))
MANUAL = {"gpt-5-4": ("2026", "year from public knowledge, exact date not verified")}
models = []
for model in sorted(set(r["model"] for r in res)):
    parts = [m.strip() for m in model.split(" + ")]
    dates = [rel.get(norm(m)) or MANUAL.get(norm(m)) for m in parts]
    known = [d for d in dates if d]
    latest = max(known, key=lambda x: x[0]) if known else (None, "unknown")
    year = int(latest[0][:4]) if latest[0] else None
    models.append({"model": model, "release_date": latest[0] if len(known) == len(parts) else None,
                   "release_year": year if len(known) == len(parts) else None, "basis": latest[1] if len(known) == len(parts) else "unknown",
                   "is_2025_2026": (year in (2025, 2026)) if len(known) == len(parts) else None,
                   "n_results": sum(1 for r in res if r["model"] == model)})

# ------------------------------------------------------------------ write
it = pd.DataFrame([{"item_key": v["item_key"], "text": v["text"], "extra": json.dumps(v["extra"], ensure_ascii=False)}
                   for v in items.values()])
assert it.item_key.is_unique
results = pd.DataFrame(res).astype({"trial": "int64", "outcome": "float64"})
dup = results.duplicated(["item_key", "model", "config", "trial"])
assert not dup.any(), results[dup].head()
assert results.item_key.isin(it.item_key).all()
it.to_parquet(f"{OUT}/items.parquet", index=False)
results.to_parquet(f"{OUT}/results.parquet", index=False)
pd.DataFrame(details).to_parquet(f"{OUT}/trial_details.parquet", index=False)
pd.DataFrame(runs).to_csv(f"{OUT}/runs.csv", index=False)
pd.DataFrame(models).to_csv(f"{OUT}/models.csv", index=False)

# ------------------------------------------------------------------ summary
print("items", len(it), it.item_key.str.split("/").str[0].value_counts().to_dict())
print("results", len(results), "runs", results.groupby(["model", "config"]).ngroups, "models", results.model.nunique())
results["version"] = results.item_key.str.split("/").str[0]
print(results.groupby("version").agg(n=("outcome", "size"), runs=("config", lambda s: s.nunique()), acc=("outcome", "mean")))
ex = it.extra.map(json.loads)
print("git identical:", collections.Counter((e["version"], e["git_instruction_identical"]) for e in ex))
print("git identical after strip:", collections.Counter((e["version"], e["git_instruction_identical_stripped"]) for e in ex))
dd = pd.DataFrame(details)
print(pd.DataFrame(models).to_string())
print("details notes:", dd.groupby([dd.run_id.str.split(":").str[0], "note"]).size().to_dict())
