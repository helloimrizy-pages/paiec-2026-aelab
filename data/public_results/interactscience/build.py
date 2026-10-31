"""Build items.parquet / results.parquet / models.csv / vlm_system_prompt.txt for InteractScience.

Usage: python -I build.py REPO OUT   (fetch.py downloads REPO and runs this)
REPO = checkout of https://github.com/open-compass/InteractScience at commit REPO_SHA below.

Sources inside REPO:
  data/interactscience.jsonl                                       150 tasks (prompt parts + metadata)
  results/lm_results/PFT_test_results/<model>_component_results.json   Playwright JSON reports, text-only setting
  results/vlm_results/component_test_results/<model>_component_results.json  same, plan + reference snapshots
  PFT_tests/<id>.spec.js                                           the Playwright tests (only used for a count check)

Outcome = the benchmark's "PFT Perfect" rule (cal_metrics.py): an item is solved when every PFT
(programmatic functional test) for that item has a result with status "passed". Timed-out tests count as
failed, as in cal_metrics.py. Nothing is re-run.
"""
import collections, csv, glob, json, os, re, sys

import pandas as pd

REPO_SHA = "c21ea8dffad175a43d810fa027a152113956e6ff"

SETTINGS = {  # report folder -> config string
    "results/lm_results/PFT_test_results": "setting=lm;prompt=lm_system_prompt+plan;temperature=0",
    "results/vlm_results/component_test_results": "setting=vlm;prompt=vlm_system_prompt+plan+reference_snapshots;temperature=0",
}

# README table of the repo (PFT Overall, PFT Average, PFT Perfect, %), text-only setting. Checked below.
README_TABLE = {
    "gpt-5": (39.47, 37.61, 16.08), "gpt-4.1": (37.07, 34.08, 11.19), "gpt-4o": (28.27, 27.09, 5.59),
    "o3": (34.93, 32.09, 13.99), "o4-mini": (37.33, 34.90, 13.29), "gemini-2.5-pro": (35.33, 34.62, 11.19),
    "gemini-2.5-flash": (31.60, 31.07, 10.49), "claude-sonnet-4-20250514": (41.47, 37.40, 13.29),
    "claude-opus-4-20250514": (40.27, 36.34, 11.19), "claude-3-5-sonnet-latest": (33.33, 31.45, 9.79),
    "deepseek-r1-0528": (33.87, 32.02, 8.39), "deepseek-v3-0324": (31.73, 30.57, 10.49),
    "kimi-k2": (31.60, 31.22, 9.79), "glm-4.5": (29.33, 26.65, 8.39), "intern-s1": (31.87, 28.93, 7.69),
    "gpt-oss-120b": (28.00, 27.78, 9.79), "gpt-oss-20b": (15.20, 12.97, 3.50),
    "Qwen3-235B-A22B-Instruct-2507": (33.33, 31.46, 13.29), "Qwen3-32B": (27.20, 24.09, 5.59),
    "Qwen3-14B": (24.13, 23.58, 7.69), "Qwen3-8B": (20.00, 18.85, 4.20), "Qwen3-4B": (14.67, 13.10, 2.80),
    "Qwen3-1.7B": (6.53, 6.22, 1.40), "Qwen2.5-Coder-32B-Instruct": (27.20, 25.10, 7.69),
    "Qwen2.5-Coder-14B-Instruct": (22.53, 20.61, 4.90), "Qwen2.5-Coder-7B-Instruct": (12.40, 10.51, 0.70),
    "Qwen2.5-VL-72B-Instruct": (23.73, 22.82, 6.99), "Qwen2.5-VL-7B-Instruct": (7.47, 6.72, 0.70),
    "Llama-3.1-70B-Instruct": (18.67, 18.04, 4.90), "Llama-3.1-8B-Instruct": (11.33, 10.16, 3.50),
}

# Public release month of the model behind each file name (vendor announcements, model cards).
# The month is approximate; the year is certain.
RELEASE = {
    "Llama-3.1-70B-Instruct": "2024-07", "Llama-3.1-8B-Instruct": "2024-07",
    "Qwen2.5-Coder-7B-Instruct": "2024-09", "Qwen2.5-Coder-14B-Instruct": "2024-11",
    "Qwen2.5-Coder-32B-Instruct": "2024-11", "Qwen2.5-VL-7B-Instruct": "2025-01", "Qwen2.5-VL-72B-Instruct": "2025-01",
    "Qwen3-1.7B": "2025-04", "Qwen3-4B": "2025-04", "Qwen3-8B": "2025-04", "Qwen3-8B-Base": "2025-04",
    "Qwen3-14B": "2025-04", "Qwen3-32B": "2025-04",
    "Qwen3-235B-A22B-Instruct-2507": "2025-07", "Qwen3-235B-A22B-Thinking-2507": "2025-07",
    "claude-3-5-sonnet-latest": "2024-10 (alias; assumed claude-3-5-sonnet-20241022)",
    "claude-opus-4-20250514": "2025-05", "claude-sonnet-4-20250514": "2025-05",
    "deepseek-r1-0528": "2025-05", "deepseek-v3-0324": "2025-03",
    "gemini-2.5-flash": "2025-06 (GA; preview 2025-04)", "gemini-2.5-pro": "2025-06 (GA; preview 2025-03)",
    "glm-4.5": "2025-07", "gpt-4.1": "2025-04", "gpt-4o": "2024-05 (alias; snapshot not stated)",
    "gpt-5": "2025-08", "gpt-oss-120b": "2025-08", "gpt-oss-20b": "2025-08", "intern-s1": "2025-07",
    "kimi-k2": "2025-07 (version not stated; 0711 or 0905)", "o3": "2025-04", "o4-mini": "2025-04",
}


def report_items(path):
    """Return {item_id: (passed, total, official)} from one Playwright JSON report, plus run start time."""
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for top in d["suites"]:
        item = top["file"]
        assert item.endswith(".spec.js"), item
        item = item[: -len(".spec.js")]
        nested, toplevel = [], []
        for spec in top.get("specs", []):          # tests written without test.describe (cal_metrics.py skips these)
            toplevel.extend(spec.get("tests", []))
        for su in top.get("suites", []):
            assert not su.get("suites"), f"deeper nesting in {path}:{item}"
            for spec in su.get("specs", []):
                nested.extend(spec.get("tests", []))
        assert not (nested and toplevel), f"mixed layout in {path}:{item}"
        tests = nested or toplevel
        if not tests:
            continue
        passed = sum(any(r.get("status") == "passed" for r in t.get("results", [])) for t in tests)
        assert item not in out, item
        out[item] = (passed, len(tests), bool(nested))
    return out, d["stats"]["startTime"], d["config"]["version"]


def main(repo, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(repo, "data/interactscience.jsonl"), encoding="utf-8")]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) == 150

    # ---- results
    res, models, per_item_tests = [], [], collections.defaultdict(set)
    for folder, config in SETTINGS.items():
        for f in sorted(glob.glob(os.path.join(repo, folder, "*_component_results.json"))):
            model = os.path.basename(f)[: -len("_component_results.json")]
            items, start, pw_version = report_items(f)
            assert set(items) == set(ids), (model, set(ids) ^ set(items))
            for item, (p, t, official) in items.items():
                per_item_tests[item].add((t, official))
                res.append(dict(item_key=item, model=model, config=config, trial=1,
                                outcome=1.0 if p == t else 0.0, pft_passed=p, pft_total=t,
                                in_official_metric=official))
            # reproduce cal_metrics.py (official items only) and compare with the README table
            off = [(p, t) for p, t, o in items.values() if o]
            overall = 100 * sum(p for p, _ in off) / sum(t for _, t in off)
            average = 100 * sum(p / t for p, t in off) / len(off)
            perfect = 100 * sum(p == t for p, t in off) / len(off)
            allp = 100 * sum(p == t for p, t, _ in items.values()) / len(items)
            setting = config.split(";")[0].split("=")[1]
            if setting == "lm" and model in README_TABLE:
                exp = README_TABLE[model]
                got = (round(overall, 2), round(average, 2), round(perfect, 2))
                assert all(abs(a - b) < 0.006 for a, b in zip(got, exp)), (model, got, exp)
            rel = RELEASE.get(model, "")
            models.append(dict(model=model, setting=setting, release=rel,
                               release_2025_2026=rel[:4] in ("2025", "2026"),
                               in_repo_readme_table=(setting == "lm" and model in README_TABLE),
                               n_items=len(items), pft_perfect_official_pct=round(perfect, 2),
                               pft_overall_official_pct=round(overall, 2), pft_average_official_pct=round(average, 2),
                               pft_perfect_all150_pct=round(allp, 2), report_start=start[:10],
                               playwright=pw_version))

    # test counts per item must be the same in every report, and match the number of test() calls in PFT_tests
    n_tests, official_flag = {}, {}
    for item, s in per_item_tests.items():
        assert len(s) == 1, (item, s)
        n_tests[item], official_flag[item] = next(iter(s))
    mism = []
    for item in ids:
        src = open(os.path.join(repo, "PFT_tests", item + ".spec.js"), encoding="utf-8").read()
        n_src = len(re.findall(r"(?<![.\w])test\s*\(", src))
        if n_src != n_tests[item]:
            mism.append((item, n_src, n_tests[item]))
    print("items whose PFT_tests test() count differs from the report:", mism)

    # ---- items
    items = []
    for r in rows:
        extra = {
            "theorem_or_principle": r["theorem or principle"], "discipline": r["discipline"],
            "difficulty": r["difficulty"], "interactive_elements_count": r["interactive elements count"],
            "plan_offset": len(r["lm_system_prompt"]),
            "vlm_images": r["image_path"], "n_pft_tests": n_tests[r["id"]],
            "pft_in_official_metric": official_flag[r["id"]],
        }
        items.append(dict(item_key=r["id"], text=r["lm_system_prompt"] + r["question"],
                          extra=json.dumps(extra, ensure_ascii=False)))
    items = pd.DataFrame(items)
    assert items.item_key.is_unique and items.text.is_unique
    assert len(set(r["lm_system_prompt"] for r in rows)) == 1 and len(set(r["vlm_system_prompt"] for r in rows)) == 1

    res = pd.DataFrame(res)
    res["trial"] = res["trial"].astype("int64")
    assert not res.duplicated(["item_key", "model", "config", "trial"]).any()
    items.to_parquet(os.path.join(out_dir, "items.parquet"), index=False)
    res.to_parquet(os.path.join(out_dir, "results.parquet"), index=False)
    pd.DataFrame(models).to_csv(os.path.join(out_dir, "models.csv"), index=False)
    with open(os.path.join(out_dir, "vlm_system_prompt.txt"), "w", encoding="utf-8") as fh:
        fh.write(rows[0]["vlm_system_prompt"])
    print(f"items {len(items)}  results {len(res)}  model-settings {len(models)}  "
          f"models {res.model.nunique()}  official-metric items {sum(official_flag.values())}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
