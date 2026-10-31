# Library sources

Each folder has `fetch.py` (downloads the pinned sources, then runs `build.py`). Outcomes follow each benchmark's own pass rule; rows without an explicit grade in the source are dropped.

**code2bench_2509**
- github.com/Code2Bench/Code2Bench @ `92fbd3da2eec702184170268fe7cb92fbaab28ec`, folders `code2bench-2509/Python` (SC-Python) and `code2bench-2509/weakly` (WSC-Python).
- Outcome: 1 when `transformed_status.json` says SUCCESS (every property-based test passes), else 0.

**cybench**
- Authors' Google Drive folder "data_by_challenges" (https://drive.google.com/drive/folders/1xkA8wdAhSSYNQERQ2B7Gpzp87qP1Wgyl). Drive has no revisions; files are fetched by the ids in `drive_manifest.csv`. One log is blocked by Drive's malware scan and has no row.
- github.com/andyzorigin/cybench @ `ca483c6443f17a20ba7d6bac3d274859a1721e51` (`task_list.txt`); github.com/cybench/cybench.github.io @ `3b62a0d0e3105cef73a1d835d603888328f5c2bb` (`data/leaderboard.csv`, cross-check only).
- Outcome: the log's own `score` on the final subtask (1 = correct flag submitted).

**icf_bench**
- github.com/qianyuli123/ICF-Bench @ `ac8149951516cf5770d78edaacc3435bb089417c`, files `<scenario>/judge_<model>.json`.
- Outcome: the pass rule in each scenario's `calculate.py`. Rows with API failures or judge parse errors are dropped.

**interactscience**
- github.com/open-compass/InteractScience @ `c21ea8dffad175a43d810fa027a152113956e6ff`: `data/interactscience.jsonl` and the Playwright test reports.
- Outcome: "PFT Perfect" from `cal_metrics.py`: 1 when every test of the item passed, else 0.

**livecodebench_pro**
- Problem statements: huggingface.co/datasets/QAQAQAQAQ/LiveCodeBench-Pro @ `adebffce047dddb7768a86bace6aea4f7425e3bc` (gated, automatic approval).
- Verdicts: the leaderboard backend (webhook.cp-bench.orzzh.com). It is not versioned, so the snapshot retrieved 2026-10-19 ships as `api_snapshot.json.gz`; `fetch_api.py` refreshes it.
- Outcome: `accepted` = 1; `not_accepted`, `compilation_error`, `generation_failed` = 0; pending verdicts get no row.

**mathvista**
- github.com/lupantech/MathVista @ `c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb` (`results/<model>/output_*.json`); huggingface.co/datasets/AI4Math/MathVista @ `2b6ad69445fbb5695c9b165475e8decdbeb97747` (testmini parquet).
- Outcome: the `true_false` field of each output file.

**multi_lcb**
- github.com/Multi-LCB/multi-lcb.github.io @ `e1f940ea1a767d46ff6ef09411cafa3917524c8a` (`data/lcb/`); github.com/Multi-LCB/Multi-LCB @ `d80be9f24268e9e5c6e1463e26c9f05417fe06a0` (prompt code, checked by SHA-256 before use); huggingface.co/datasets/livecodebench/code_generation_lite @ `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` (statements, streamed; test cases dropped).
- Outcome: the site's `pass` value, the fraction of samples that passed every test.

**nyu_ctf_bench**
- github.com/NYU-LLM-CTF/leaderboard_submissions @ `69def02a79e8b1d890e1e0d5c11106a9f40e11e1` (`transcripts/*/summary.json`); github.com/NYU-LLM-CTF/NYU_CTF_Bench @ `4c5744f55a31a58e03d6118c9dbb1c18e5bbdfa9` (challenge texts).
- Outcome: `results[challenge]` true = 1, false = 0; only the 200 test challenges.

**open_proof_corpus**
- huggingface.co/datasets/INSAIT-Institute/OPC @ `dcc3b4804e2d126ea34b13e3e0cd998c3302644b`.
- Outcome: mean of the human 0/1 grades in `score`; contradictory rows are dropped.

**osworld**
- huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs @ `5473c39e42a538a187a9b2c2b499db59d560fd8c` (small result files, read from the zips by range requests); github.com/OS-World/OS-World.github.io @ `62f8466dbe8b4d67c104b5ead3f8271da01aa09d` (`static/data/osworld_verified_results.xlsx`); github.com/xlang-ai/OSWorld @ `9f4b4d34319589f3ba6947da6045b0955b08900e` (task-file history).
- Google Drive folder `1XtGJe3NYbceLh58qHqXId81_d5rwFLwQ`, linked from the leaderboard. Not versioned; the 54 files used are fetched by the ids in `drive_manifest.csv`.
- Outcome: OSWorld's own task score in [0, 1] from `result.txt` (fractional scores kept). A few runs released only `all_result.json` or `task_summary.json`; those files are used for them.

**swe_bench_live**
- github.com/SWE-bench-Live/submission @ `641b727c8a5e3e5d0af08a348e632b12c3f11aef`; HF SWE-bench-Live/SWE-bench-Live @ `b51a86422e10cfd403beb4773e5a2947953e36ec`, SWE-bench-Live/MultiLang @ `3638632e8153a10ca422c1022bed79023084b5c9`, SWE-bench-Live/Windows @ `b9fa25f2732dbc2c93b9738b610d053d8b7bd7a9`.
- Outcome: resolved or success = 1, unresolved or failure = 0. Error, incomplete, empty-patch and conflicting ids get no row.

**swe_bench_pro**
- github.com/scaleapi/SWE-bench_Pro-os @ `66f92766bba642462d4bbe5479e83f91f9211862` (9 runs); HF ScaleAI/SWE-bench_Pro `external_hf.csv` @ `9c13b199fe2d3195214e2f0c6bf3c7d6f81e3877` and @ `8555f71a4f12cbf659c8f242769714168586cc85`, configs @ `2d52cb3df914a3fcf80c7f66738b3a88ae37fc50`; one `eval_results.json` from Scale's public S3 bucket (checked by SHA-256).
- Outcome: `eval_results.json` true (all fail-to-pass and pass-to-pass tests pass) = 1, false = 0.

**terminal_bench**
- Harbor Hub public read API (not versioned). The Hub data retrieved 2026-10-21 ships as `hub_snapshot.tar.xz` and is used by default; `fetch.py --live` fetches the current Hub.
- HF harborframework/terminal-bench-2-leaderboard @ `572b2614be2c0cb2527e14f5b1e4026f1072e6c1`; github.com/harbor-framework/terminal-bench @ `2e29ca91d5f1cb6fb60924bf867acc755565c044` (tags v3.0.0, v4.0.0), terminal-bench-2-1 @ `7131e4375048a0e408a8fb404b5f499d726b695b`, terminal-bench-2 @ `2fd12b88aafdd04a52c298e3940bcb189f9766d6`.
- Outcome: 1 if reward > 0, else 0. Errored trials get no row. In TB 2.1, when all successes of a (run, task) pair were disqualified for reward hacking they count as 0; when only some were, the pair is dropped.
