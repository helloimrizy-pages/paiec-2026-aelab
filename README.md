# AELab: PAIEC 2026 entry

Code and data for team AELab's entry to the NeurIPS 2026 Predictive AI Evaluation Competition (PAIEC). The program predicts the probability that an AI system answers a benchmark item correctly, without running the system. This repository holds everything needed to rerun the submitted program and the results in our technical report.

Our final Codabench submission is run **1003836**. `submissions/v9_lib/` holds its program. Apart from one comment line in `publib.py`, every file equals the zip of that run.

## Method (v9)

- An LLM judge (gpt-5.6-luna, low reasoning effort) estimates how likely the system is to solve the item.
- Two models (gpt-5.6-luna and Claude Haiku 5.5, medium effort) attempt the item. Only the length of each attempt is kept, as an effort score that ranks items within a system–benchmark pair.
- A per-pair Bayesian logistic model with a between/within (Mundlak) split combines the two signals and updates on the revealed labels.
- When the item text matches an item in our library of public per-item results, a lookup rule mixes the recorded outcomes with an item-easiness model instead.

## Layout

| Path | Contents |
|---|---|
| `submissions/v9_lib/` | the submitted program (`model.py`) and its library index (`publib_data/`) |
| `lab/` | protocol simulator, data loader, ranking model and library builder |
| `lab/features/` | cached judge and attempt outputs for the training benchmarks |
| `eval/` | calibration and analysis scripts behind the report's tables, and the offline check |
| `eval/results/` | shipped inputs of the `eval/` scripts (their outputs are written here too) |
| `data/public_results/` | download and build scripts for the 13 library benchmarks; `SOURCES.md` lists every source |
| `data/mdb_profile.csv` | per-benchmark profile of the larger measurement-db checkpoint (used by `eval/h21.py diag`) |
| `tools/` | library builder and submission zip builder |

## Setup

We used Python 3.12 on macOS (Apple M4 Max). Everything below except the library downloads runs on a CPU in minutes.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The training data (measurement-db) is gated. Request access at https://huggingface.co/datasets/aims-foundations/measurement-db, log in with `hf auth login`, then download the six competition benchmarks at the pinned revision (about 65 MB):

```
python lab/fetch_data.py core
```

## 1. Rerun the submitted program

Build the submission zip with your own API keys:

```
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python tools/build_zip.py v9_lib
```

- The script first checks both keys with free model-list requests. It then writes `dist/v9_lib_private.zip` (keys inside; upload this one to Codabench) and `dist/v9_lib_public.zip` (no keys). `ANTHROPIC_WORKSPACE_ID` is optional.
- At scoring time the program calls gpt-5.6-luna through the OpenAI Responses API and claude-haiku-5-5 through the Anthropic Messages API. It uses no other network access.
- From the per-item costs of our development batches at standard prices, one hidden-test run of 1,000 subject–item combinations costs up to about $6.40 in API calls.
- API outputs are not bitwise reproducible, so a rerun gives slightly different predictions. A single hidden-test draw also has a standard deviation of about 0.02 in Brier ALC, so a run on a new draw will not match our scores exactly.

## 2. Offline check (no API calls)

```
cd lab
python ../eval/check_v9_equivalence.py
```

The check refits the calibration from the cached judge and attempt outputs and asserts that it equals `submissions/v9_lib/judge_config.json`. It then compares 432 predictions of the deployed `model.py` with the research implementation. It ends with `"status": "passed"`, with a largest difference of about 1e-15. Its library part runs only after the library is built (section 4).

## 3. Reproduce the report's results

Run every command from `lab/` with the environment active. None of them calls an API. Times are wall-clock times on one Apple M4 Max laptop.

| Result in the report | Command | Time |
|---|---|---|
| Development ladder: v3, v4, v5 | `python ranking_model.py --variants v3,v4,adaptive_total --splits 20 --seed 5 --tag primary` | 14 s |
| Development ladder: v5 interval (validation seed) | `python ranking_model.py --variants v3,v4,adaptive_total --splits 40 --seed 20261005 --tag validation` | 24 s |
| Development ladder: v7 (row `GA`) | `python ../eval/combo.py 5 20 v7primary v5,G,GA` | 64 s |
| Development ladder: v8; the refit variants of the level calibration | `python ../eval/h21.py pilot 5 20 h21all pilot,ckpt_head,ckpt_level,ckpt_both,ckpt_lw,both_both,blend50` | 287 s |
| Level error on the 97 checkpoint benchmarks (0.273 to 0.205) | `python ../eval/h21.py diag` | 2 s |
| Short episodes on the 97 checkpoint benchmarks (refit and v8 gains) | `python ../eval/h21c.py` | 11 s |
| Development ladder: v9 core; Opus as a third attempter | `python ../eval/h23_ens.py 5 20 primary luna,luna+haiku,luna+haiku+opus` | 113 s |
| Within-pair AUC of each attempter's effort score | `python ../eval/attempt_auc.py luna,haiku,sonnet,opus,luna+haiku` | 2 s |
| Library simulation (all regimes and budgets) | `python ../eval/h22_sim.py 5 20` | 48 s |
| More calibration benchmarks for budget 0 (negative result) | `python ../eval/h18.py pilot 5 20 h18primary` | 93 s |
| Matcher recall and false matches (needs section 4 and `python lab/fetch_data.py full`) | `python h22_library.py selftest` | 6 s |

Notes:

- In the `eval/` scripts, the two numbers are the seed and the number of acquisition splits per pair. The next argument is a free tag for the output files, and any further argument lists the variants to compare.
- For v4, `ranking_model.py` gives the reported ALC and change (0.1750 and −0.0032) with the interval [−0.0058, −0.0000]. The interval in the report came from an earlier development script and differs in the fourth decimal.
- `python lab/fetch_data.py full`, run from the repository root, downloads the item tables of 59 benchmarks of the larger measurement-db checkpoint (revision 54074453, about 425 MB). Only the false-match part of the self-test uses them.

Results that this repository cannot rerun:

- v1, v2 and the two ladder steps between v2 and v3, and the step that used the reasoning tokens of one attempt;
- the simulator checks against the hidden scores, and the analysis of score against success rate in Appendix A of the report;
- the negative results other than the extra calibration benchmarks, the oracle analyses, and the false-match scan of 57,796 checkpoint items.

These ran in earlier development code that depends on further caches. The hidden-test scores come from the competition platform and cannot be rerun offline.

## 4. Rebuild the library

`submissions/v9_lib/publib_data/` already holds the library index. To rebuild it from the benchmark providers' own releases:

```
python tools/build_library.py            # all 13 benchmarks in data/public_results/READY.txt, then the index
python tools/build_library.py --export   # the same, and also writes the new index into submissions/v9_lib/publib_data/
```

- A full run takes about 80 minutes. It keeps 2.4 GB of downloads in `data/downloads/` and transfers about 26 GB, most of it streamed and not stored. It needs `git`.
- Each folder in `data/public_results/` has a `fetch.py` that downloads that benchmark's sources at pinned revisions and then runs its `build.py`. `SOURCES.md` lists every source, its revision and the pass rule.
- Four sources have no versions. For the two Google Drive folders we ship the file-id lists (`cybench/drive_manifest.csv`, `osworld/drive_ids.txt`, `osworld/drive_manifest.csv`). For the LiveCodeBench Pro leaderboard backend and the Terminal-Bench Harbor Hub API we ship snapshots (`livecodebench_pro/api_snapshot.json.gz`, 19 October 2026; `terminal_bench/hub_snapshot.tar.xz`, 21 October 2026).
- LiveCodeBench Pro is gated on Hugging Face. Request access once at https://huggingface.co/datasets/QAQAQAQAQ/LiveCodeBench-Pro (approval is automatic).
- Model names are mapped to measurement-db canonical names with `map_model_registry.json` from aims-foundations/measurement-db at a pinned commit, downloaded and checked by hash.
- From the per-benchmark tables, the index step alone (`cd lab && python h22_library.py build`, then `python h22_library.py export <dir>`) reproduces the shipped index byte for byte.

## Data

- **measurement-db** (revisions `bc8204d8` and `54074453`) is gated and is not redistributed here. `lab/fetch_data.py` downloads it.
- **Cached model outputs** in `lab/features/` and `eval/results/ext/` hold judge outputs (probabilities, ratings and a short benchmark guess) and attempt token counts for measurement-db items, keyed by item id. Some files also keep the short final answers of earlier attempts. No item text is included.
- **The library index** in `submissions/v9_lib/publib_data/` is derived from the public per-item results of the 13 benchmarks in `data/public_results/READY.txt`, all listed in the competition's benchmark inventory. It stores hashed text shingles, never item text, and the recorded outcomes per model and configuration. Each source keeps its own licence.
- We never used hidden-test feedback to choose library benchmarks or to identify hidden benchmarks or subjects. No feedback files are part of this repository.

## Licence

The code in this repository is released under the MIT License (see `LICENSE`). Data derived from third-party releases keeps the licence of its source. This covers the library index in `submissions/v9_lib/publib_data/` and the snapshots and file lists in `data/public_results/`; `data/public_results/SOURCES.md` lists every source.
