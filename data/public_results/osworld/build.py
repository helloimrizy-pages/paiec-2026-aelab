#!/usr/bin/env python3
"""Build the OSWorld per-item public results: items.parquet, results.parquet, runs.csv, models.csv, build_summary.json.

Usage:  python -I build.py <dl_dir> <out_dir>      (fetch.py fills <dl_dir> and then runs this script)

Sources in <dl_dir> (all OSWorld team releases):
  HF dataset xlangai/ubuntu_osworld_verified_trajs @ 5473c39e42a538a187a9b2c2b499db59d560fd8c
    https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs
    listings/<zip>.json, members/<zip>/members.json        zip central directories and small result files (HTTP range requests)
    listings/holo3_internal_<t>.json, members/holo3_internal_<t>/members.json   zips nested in a stored zip
    members/oswv_claude-sonnet-4-6.tar.gz/, members/coact1-150-100-50-15steps.zip/  streamed archives (members.json + tar_index.json)
  OS-World/OS-World.github.io @ 62f8466dbe8b4d67c104b5ead3f8271da01aa09d   https://github.com/OS-World/OS-World.github.io
    site/osworld_verified_results.xlsx                     verified leaderboard (static/data/)
  Google Drive folder 1XtGJe3NYbceLh58qHqXId81_d5rwFLwQ (linked from the leaderboard page; not versioned, ids in drive_manifest.csv)
    drive/listings/<id>.json, drive/members/<id>/members.json, drive/small/<id>.bin (three all_result.json files, one R-MCTS tar.gz)
  xlang-ai/OSWorld @ 9f4b4d34319589f3ba6947da6045b0955b08900e   https://github.com/xlang-ai/OSWorld
    repo/osworld_full.git                                  bare clone with the task files of the first-parent history
Outcome: OSWorld's own task score in [0, 1] as written to result.txt by env.evaluate() (True/False -> 1.0/0.0; no grade -> no row);
  all_result.json for CoAct-1 and three Drive runs that released only that file; task_summary.json reward for Holo3 internal runs.
Item text: the task file's instruction at the main-branch commit of the run's start day; one item per distinct text of a task.
Only small result files are downloaded. Nothing from a download is executed.
"""
import ast, collections, datetime, gzip, io, json, os, re, subprocess, sys, tarfile
import pandas as pd

DL, OUT = sys.argv[1], sys.argv[2]
REPO = os.path.join(DL, "repo", "osworld_full.git")
HEAD_SHA = "9f4b4d34319589f3ba6947da6045b0955b08900e"
HF_REV = "5473c39e42a538a187a9b2c2b499db59d560fd8c"
SITE_SHA = "62f8466dbe8b4d67c104b5ead3f8271da01aa09d"
DOMAINS = {"chrome", "gimp", "libreoffice_calc", "libreoffice_impress", "libreoffice_writer",
           "multi_apps", "os", "thunderbird", "vlc", "vs_code"}
VERIFIED_LAUNCH = "2025-07-28"   # OSWorld-Verified announcement / first leaderboard date
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def parse_score(txt):
    """Explicit grade written by OSWorld's evaluator into result.txt. None = no grade."""
    s = (txt or "").strip().strip("\x00").strip()
    if s in ("True", "False"):
        return 1.0 if s == "True" else 0.0
    try:
        x = float(s)
    except ValueError:
        return None
    return x if 0.0 <= x <= 1.0 else None


def task_of(path):
    """(domain, task_id) if path is <...>/<domain>/<uuid>/<file> and file is a top-level task file."""
    p = path.split("/")
    if len(p) >= 3 and p[-3] in DOMAINS and UUID.match(p[-2]):
        return p[-3], p[-2], "/".join(p[:-3])
    return None


def zip_date(listing, name):
    d = listing.get(name)
    return "%04d-%02d-%02d" % tuple(d[:3]) if d else None


def load_json_any(s):
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)


# ----------------------------------------------------------------------------- leaderboard
lb = pd.read_excel(os.path.join(DL, "site", "osworld_verified_results.xlsx"))


def excel_date(x):
    if isinstance(x, (int, float)) and not pd.isna(x):
        return (datetime.date(1899, 12, 30) + datetime.timedelta(days=int(x))).isoformat()
    if pd.isna(x):
        return None
    return str(x)[:10]


LB = {}
for i, r in lb.iterrows():
    m = re.match(r"\s*([\d.]+)\s*/\s*(\d+)", str(r["Success/Total"]))
    LB[i] = dict(model=str(r["Model"]).strip(), institution=r["Institution"], approach=r["Approach type"],
                 max_steps=None if pd.isna(r["Max steps"]) else int(r["Max steps"]),
                 a11y=r["Additional a11y tree used"], coding=r["Additional coding-based action"],
                 multi=r["Multiple rollout"], date=excel_date(r["Date"]),
                 lb_sum=float(m.group(1)) if m else None, lb_n=int(m.group(2)) if m else None)

# ----------------------------------------------------------------------------- run catalogue
# HF verified runs: (zip, root substring or "" for the only root, leaderboard row, extra dict)
# Leaderboard rows were matched by (score sum, n) and checked below; see runs.csv.
HF = [
    ("OSWorldSurfer-Holo3-35B-A3B-20260330-OSWorld-Verified_20260420_verified-run_with-local-rewards.zip",
     ["results_hcompany_verified_run1_20260420_121655", "results_hcompany_verified_run1_repair_20260420_183538"], 137,
     dict(note="maintainer-run verification; run1 merged with its 2-task repair run as in merged_summary_with_repair.json")),
    ("OSWorldSurfer-Holo3-35B-A3B-20260330-OSWorld-Verified_20260420_verified-run_with-local-rewards.zip",
     ["results_hcompany_verified_run2_20260420_191308"], 138, dict(note="maintainer-run verification")),
    ("UI-TARS-0717-100step.zip", [""], 18, dict(flags=["label_disputed"], note="zip named UI-TARS-0717; args.model is an endpoint id; leaderboard row says doubao-1-5-thinking-vision-pro-250717")),
    ("UI-TARS-0717-15step.zip", [""], 17, dict(flags=["label_disputed"], note="zip named UI-TARS-0717; leaderboard row says doubao-1-5-thinking-vision-pro-250717")),
    ("autoglm_15steps.zip", [""], 85, {}), ("autoglm_50steps.zip", [""], 86, {}),
    ("claude-3-7-sonnet-20250219-15steps.zip", [""], 9, {}), ("claude-3-7-sonnet-20250219-50steps.zip", [""], 10, {}),
    ("claude-3-7-sonnet-20250219-100steps.zip", [""], 11, {}),
    ("claude-4-sonnet-20250514-15steps.zip", [""], 12, {}), ("claude-4-sonnet-20250514-50steps.zip", [""], 13, {}),
    ("claude-4-sonnet-20250514-100steps.zip", [""], 14, {}),
    ("claude-sonnet-4-5-20250929_15steps.zip", [""], 119, {}), ("claude-sonnet-4-5-20250929_50steps.zip", [""], 120, {}),
    ("claude-sonnet-4-5-20250929_100steps.zip", [""], 121, {}),
    ("doubao-1-5-thinking-vision-pro-250428-15step.zip", [""], 15, {}),
    ("doubao-1-5-thinking-vision-pro-250428-100step.zip", [""], 16, {}),
    ("evocua_20260105.zip", [""], 128, {}), ("evocua_8b_20260105.zip", [""], 129, {}),
    ("intelligence-indeed-claude-sonnet-4-6-opus-4-7-nogdrive-100steps.zip", [""], 146, dict(note="args.model claude-opus-4-7; zip name says sonnet-4-6 + opus-4-7")),
    ("jedi-7b-4o-15steps.zip", [""], 69, {}), ("jedi-7b-4o-50steps.zip", [""], 70, {}), ("jedi-7b-4o-100steps.zip", [""], 71, {}),
    ("jedi-7b-o3-15steps.zip", [""], 72, {}), ("jedi-7b-o3-50steps.zip", [""], 73, {}), ("jedi-7b-o3-100steps.zip", [""], 74, {}),
    ("kimi-k25.zip", [""], 131, dict(note="357 result files; leaderboard shows n=361")),
    ("kimi-k26.zip", [""], 135, {}),
    ("kimi-vl-a3b-15step.zip", [""], 61, {}), ("kimi-vl-a3b-100step.zip", [""], 62, {}),
    ("mobile-agent-v3-gui-owl-7b.zip", [""], 87, {}), ("mobileagent_v3.zip", [""], 88, {}),
    ("o3_15steps.zip", [""], 6, {}), ("o3_50steps.zip", [""], 7, {}), ("o3_100steps.zip", [""], 8, {}),
    ("o3_gta1_50steps.zip", [""], 75, {}), ("o3_gta1_100steps.zip", [""], 76, {}),
    ("opencua_agent-opencua_32b-cot_l2-action_history-3image-Ubuntu-15steps.zip", ["turn_1"], 43, {}),
    ("opencua_agent-opencua_32b-cot_l2-action_history-3image-Ubuntu-15steps.zip", ["turn_2"], 44, {}),
    ("opencua_agent-opencua_32b-cot_l2-action_history-3image-Ubuntu-15steps.zip", ["turn_3"], 45, {}),
    ("opencua_agent-opencua_32b-cot_l2-action_history-3image-Ubuntu-50steps.zip", ["turn_1"], 46, {}),
    ("opencua_agent-opencua_32b-cot_l2-action_history-3image-Ubuntu-50steps.zip", ["turn_2"], 47, {}),
    ("opencua_agent-opencua_32b-cot_l2-action_history-3image-Ubuntu-50steps.zip", ["turn_3"], 48, {}),
    ("opencua_agent-opencua_32b-cot_l2-action_history-3image-Ubuntu-100steps.zip", ["turn_1"], 49, {}),
    ("opencua_agent-opencua_32b-cot_l2-action_history-3image-Ubuntu-100steps.zip", ["turn_2"], 50, {}),
    ("opencua_agent-opencua_32b-cot_l2-action_history-3image-Ubuntu-100steps.zip", ["turn_3"], 51, {}),
    ("opencua_agent-opencua_7b-cot_l2-action_history-3image-Ubuntu-15steps.zip", ["turn_1"], 34, {}),
    ("opencua_agent-opencua_7b-cot_l2-action_history-3image-Ubuntu-15steps.zip", ["turn_2"], 35, {}),
    ("opencua_agent-opencua_7b-cot_l2-action_history-3image-Ubuntu-15steps.zip", ["turn_3"], 36, {}),
    ("opencua_agent-opencua_7b-cot_l2-action_history-3image-Ubuntu-50steps.zip", ["turn_1"], 37, {}),
    ("opencua_agent-opencua_7b-cot_l2-action_history-3image-Ubuntu-50steps.zip", ["turn_2"], 38, {}),
    ("opencua_agent-opencua_7b-cot_l2-action_history-3image-Ubuntu-50steps.zip", ["turn_3"], 39, {}),
    ("opencua_agent-opencua_7b-cot_l2-action_history-3image-Ubuntu-100steps.zip", ["turn_1"], 40, {}),
    ("opencua_agent-opencua_7b-cot_l2-action_history-3image-Ubuntu-100steps.zip", ["turn_2"], 41, {}),
    ("opencua_agent-opencua_7b-cot_l2-action_history-3image-Ubuntu-100steps.zip", ["turn_3"], 42, {}),
    ("opencua_agent-opencua_a3b-cot_l2-action_history-3image-Ubuntu-15step.zip", [""], 28, {}),
    ("opencua_agent-opencua_a3b-cot_l2-action_history-3image-Ubuntu-50step.zip", [""], 29, {}),
    ("opencua_agent-opencua_a3b-cot_l2-action_history-3image-Ubuntu-100step.zip", [""], 30, {}),
    ("opencua_agent-opencua_qwen2_7b-cot_l2-action_history-3image-Ubuntu-15step.zip", [""], 31, {}),
    ("opencua_agent-opencua_qwen2_7b-cot_l2-action_history-3image-Ubuntu-50step.zip", [""], 32, {}),
    ("opencua_agent-opencua_qwen2_7b-cot_l2-action_history-3image-Ubuntu-100step.zip", [""], 33, {}),
    ("qwen2.5-vl-32b-instruct_15step.zip", [""], 57, {}), ("qwen2.5-vl-32b-instruct_100step.zip", [""], 58, {}),
    ("qwen2.5-vl-72b-instruct_15step.zip", [""], 59, {}), ("qwen2.5-vl-72b-instruct_100step.zip", [""], 60, {}),
    ("qwen35_osworld_eval_traj.zip", [""], None, dict(
        model="Qwen3.5-Plus (internal checkpoint qwen35_plus_nothink_cpt_gemini_s1_0213)", institution="Qwen Team, Alibaba Group",
        approach="General model", max_steps=100, flags=["not_on_leaderboard"],
        note="no leaderboard row; model label taken from the result path (h100 = 100 steps); no args.json")),
    ("qwen37-plus_100steps_0524.zip", [""], 142, {}),
    ("result_aws_dart_gui_20260122.zip", [""], 130, {}),
    ("results-vlaa-gui.zip", [""], 139, dict(flags=["label_disputed"], note="leaderboard says Opus 4.5; args.model is global.anthropic.claude-opus-4-6-v1")),
    ("results_UI-TARS-2-2509_100steps.zip", [""], 116, {}),
    ("results_agent_s2_gemini_15steps.zip", [""], 63, {}), ("results_agent_s2_o3_15steps.zip", [""], 65, {}),
    ("results_agi0_50steps.zip", [""], None, dict(model="AGI-0", institution="AGI Company", approach="Agentic framework", max_steps=50,
                                                  flags=["not_on_leaderboard"], note="no leaderboard row")),
    ("results_agi0_50steps_1117.zip", [""], None, dict(model="AGI-0", institution="AGI Company", approach="Agentic framework", max_steps=50,
                                                       flags=["not_on_leaderboard"], note="no leaderboard row; second run dated 2025-11-17")),
    ("results_autoglm_v_15steps.zip", [""], 96, {}), ("results_autoglm_v_50steps.zip", [""], 97, {}),
    ("results_bbon_10_72_6.zip", [""], 110, dict(flags=["multiple_rollout"], note="Behavior Best-of-N over 10 rollouts; one selected outcome per task")),
    ("results_fable5_task218872_osworld1.0_shareable.zip", [""], 148, dict(model="claude-fable-5[1m]", flags=["label_disputed"],
        note="result folder and zip name say claude-fable-5[1m]; the leaderboard row with this exact score (300.2/360) is labelled claude-opus-5[1m]")),
    ("results_full361_adaptive_20260609_094019.zip", [""], 143, dict(note="byte-identical duplicate results_full361_adaptive_20260609_094019_minimax_m3.zip skipped")),
    ("results_gbox_15steps.zip", [""], 122, {}),
    ("results_gemini_50_steps_aws.zip", [""], 64, {}),
    ("results_hippo_agent.zip", [""], 132, {}),
    ("results_klick_openapa_full_newkey_bg2.zip", [""], 136, {}),
    ("results_maestro.zip", [""], 89, {}), ("results_maestro_50steps_2_results_only.zip", [""], 90, {}),
    ("results_maestro_100steps_results_only.zip", [""], 91, {}),
    ("results_mano_100steps.zip", [""], 94, {}),
    ("results_muse_spark_super_nova_ext_xhigh_100steps_full361_20260709_175646_sanitized.zip", [""], 145, dict(note="agent super_nova_ext, reasoning effort xhigh (from file name)")),
    ("results_o3_tars1p5_50_steps.zip", [""], 66, {}), ("results_o3_tars1p5_100_steps.zip", [""], 67, {}),
    ("results_omnitars_100_steps.zip", [""], 92, {}),
    ("results_opus4p5_single_agent.zip", [""], 111, dict(note="one of the five N=1 rows; matches row with 243.52/361")),
    ("results_opus5_task221874_osworld1.0_shareable.zip", [""], 147, dict(model="claude-opus-5[1m]", flags=["label_disputed"],
        note="result folder and zip name say claude-opus-5[1m]; the leaderboard row with this exact score (307.73/358) is labelled claude-fable-5[1m]")),
    ("results_seed18.zip", [""], 123, {}),
    ("results_tianxi_action_7b_turn1.zip", [""], 83, {}), ("results_tianxi_action_7b.zip", [""], 84, {}),
    ("uipath_gpt_5_50steps.zip", [""], 93, {}),
    ("uitars-72b-dpo-15step.zip", [""], 25, {}), ("uitars-72b-dpo-50step.zip", [""], 26, {}), ("uitars-72b-dpo-100step.zip", [""], 27, {}),
    ("uitars15-7b-15step.zip", [""], 19, {}), ("uitars15-7b-15step-1.zip", [""], 20, {}),
    ("uitars15-7b-50step.zip", [""], 21, {}), ("uitars15-7b-50step-1.zip", [""], 22, {}),
    ("uitars15-7b-100step.zip", [""], 23, {}), ("uitars15-7b-100step-1.zip", [""], 24, {}),
    ("oswv_claude-sonnet-4-6.tar.gz", [""], 133, dict(note="358 result files; leaderboard shows 256.71/356")),
    ("coact1-150-100-50-15steps.zip", [""], 80, dict(score_file="all_result.json",
        note="archive holds one run (all_result.json sum 219.34/361 = the 150-step row); one task where result.txt and all_result.json disagree is dropped")),
]
HF_SKIP = {
    "qwen36-cua-nothink_300steps_0510.zip": "OSWorld-V2 tasks (different benchmark version)",
    "qwen36-cua-think_300steps_0510.zip": "OSWorld-V2 tasks (different benchmark version)",
    "results_opus5_task220297_osworld2.0_shareable.zip": "OSWorld 2.0 tasks (different benchmark)",
    "results_full361_adaptive_20260609_094019_minimax_m3.zip": "byte-identical duplicate of results_full361_adaptive_20260609_094019.zip",
    "OSWorldSurfer-Holo3-35B-A3B-20260330-OSWorld-Verified_hcompany-internal-runs-20260416_complete-with-rewards.zip": "read through its three nested zips (holo3_internal_*)",
}
HOLO_INTERNAL = ["072452", "072955", "073458"]

# Drive (provider folder linked from the leaderboard page; original OSWorld, self-reported table)
# id -> (model, setting, max_steps, flags, note); setting = observation type in the self-reported table
DRIVE = {
    "1miMp-AR_hOfLUWszGb_aZG2f_Nd66Ntt": ("GPT-4 Vision (gpt-4-vision-preview)", "screenshot", 15, [], ""),
    "17jgZEjslxIZPC5kuzOq6oTK5EuDiJ-_l": ("Gemini-Pro Vision", "screenshot", 15, [], ""),
    "1mN12zjCiawfU4k1ifPsLW-hcuzPV7Voz": ("GPT-4o", "screenshot", 15, [], ""),
    "1Skq_uMcAR08LMDnIpOswjxm5ntbVXm5M": ("GPT-4o-mini", "screenshot", 15, [], ""),
    "1KSYs7ZOhEkjBQX5UnMpLkbxZZ6BLhbDm": ("InternVL2", "screenshot", 15, [], ""),
    "1geppLqwV80p66bXdffVBbLayjCkIQ9mk": ("Llava-OneVision", "screenshot", 15, [], "zip and folder named llava-next"),
    "1T58G_EIr4pWZYJ8plXEWspFHr35g-Nsy": ("MiniCPM-V 2.6", "screenshot", 15, [], ""),
    "1XJiHGMJvExo0c_kFI6SRGpoGBPvInLDR": ("GPT-4 (gpt-4-0125-preview)", "a11y_tree", 15, [], ""),
    "1EOt8qzzQT0djkRZkTcVC7t1UvOPRyyyx": ("GPT-4o", "a11y_tree", 15, [], ""),
    "1ezGgCb05ZH6Y8fakQ6m4yQiizy-5KTW5": ("GPT-4 Vision (0409) (gpt-4-turbo-2024-04-09)", "a11y_tree", 15, [], ""),
    "1F-nKdsxC87wk1dc4Eau7xwVSyozA8Vdn": ("Qwen-Max", "a11y_tree", 15, [], "the table links this file for both Qwen-Max and Qwen-vl-Max-0809 (a11y); folder and score match Qwen-Max"),
    "1msv8AcdvAR1cm1aDOi_HArlgTKzcYGys": ("GPT-4o-mini", "a11y_tree", 15, [], ""),
    "1a78Tg0MguATou3e8pIO02PF0tqqznXKB": ("Gemini-Pro-1.5", "a11y_tree", 15, [], ""),
    "1qEY7LNc00qnAoAcS5kzPX6WkAp-tqz0s": ("Llava-OneVision", "a11y_tree", 15, [], "zip named llava-next"),
    "19llLwbfmjJIAACxpbXVKdRdvITsLhcG8": ("GPT-3.5 (gpt-3.5-turbo-16k)", "a11y_tree", 15, [], ""),
    "1QdYgLmpCXNNoBD07XxW-rFHuRtv7sZPp": ("Gemini-Pro", "a11y_tree", 15, [], ""),
    "10AL-m6I_h_GYJMfPIl9lk8i2FOH8k9Bb": ("Llama-3-70B", "a11y_tree", 15, [], ""),
    "16H44qC5n2yz68po-DSifqsFkmLnqp8rU": ("GPT-4 Vision (gpt-4-vision-preview)", "screenshot_a11y_tree", 15, [], ""),
    "1B4N6Hog3SuoTV2KL21hwANgdDbex4Oh8": ("GPT-4o", "screenshot_a11y_tree", 15, [], ""),
    "1rQelSpJAtiyQz6V9CK4ZrUXKIwvMqi_S": ("GPT-4 Vision (0409) (gpt-4-turbo-2024-04-09)", "screenshot_a11y_tree", 15, [], ""),
    "1fXNamXtbXwZKiSOhoAEUmHQoJ3biOffQ": ("GPT-4o-mini", "screenshot_a11y_tree", 15, [], ""),
    "1lxDcOyQpi0Ysiw2HIowSb2-slyY5KBY8": ("Gemini-Pro-1.5", "screenshot_a11y_tree", 15, [], ""),
    "1KyRWM5xrI0lW_hUqgjpKmA-xfNSk5z_1": ("Qwen-vl-Max-0809", "screenshot_a11y_tree", 15, ["partial_run"], "only 214 of 369 tasks have a result file"),
    "1YFsNMKTzB1Cgo4xWexw2syFjLoxSNUBx": ("Gemini-Pro Vision", "screenshot_a11y_tree", 15, [], ""),
    "1-32Vw7fRRJYrYXS5mLGb-0dAz96e750D": ("Llava-OneVision", "screenshot_a11y_tree", 15, [], "zip named llava-next"),
    "1svJtLg96MFByPM21xin6q3hL5Ksics2W": ("GPT-4 Vision (gpt-4-vision-preview)", "som", 15, [], ""),
    "1hW-4TPo1pyN1AiYnYRnQx4MhANzilKpY": ("GPT-4 Vision (0409) (gpt-4-turbo-2024-04-09)", "som", 15, [], ""),
    "1M6LG3EMiZrQE5nLu2EmuMctLnbmRrSIo": ("Claude-3-Opus (claude-3-opus-20240229)", "som", 15, [], ""),
    "1G59RtEDg4-jVEjUwHlDV1NKvPQA9antF": ("GPT-4o", "som", 15, [], ""),
    "15O6B2poXnN6VxHVOyJCjv94Ltx9qnVJi": ("GPT-4o-mini", "som", 15, [], ""),
    "1qlw5ejuNw_q6gf4iz3Vycc3QL8ArCVkY": ("Llava-OneVision", "som", 15, [], "zip named llava-next"),
    "1FP7U6QlGn1NUqQucLnUzKy_IfKotYQb1": ("Gemini-Pro Vision", "som", 15, [], ""),
    "1M-PAz9S7cvYR8T6-VIEZKMbw7yVTN5je": ("InfantAgent (Claude-3.7 Sonnet planner + UI-TARS-1.5-7B grounder)", "screenshot", 50, ["self_reported"], ""),
    "1PpuM2-EHlRAJ6QzD3ET6zYH9_D0S9x3x": ("Jedi-3B w/ gpt-4o", "screenshot", 100, ["self_reported"], "folder jedi-3b-100steps-osw-trajs"),
    "18XTQE8orPu2Fh8o4A3QsJMy9rxlTOcSa": ("Jedi-7B w/ gpt-4o", "screenshot", 100, ["self_reported"], "folder jedi-7b-100steps-osw-trajs"),
    "16y7s_8dX6O1Idg5glrDb8AsITn40QPsv": ("TianXi-Action-7B", "screenshot", None, ["self_reported"], "TianXi_Action_7B-Lenovo_Research_PCIE_Lab_0618.zip (June 2025 submission, before OSWorld-Verified)"),
    # Agent S2 / S2.5 folder (Simular). s2_5 zips for o3 15/50/100 and gemini 15 are identical to HF zips -> HF copy kept.
    "14mFPz5iYrpn3ET3b_fI_VffxISSPrGFw": ("agent s2 w/ gemini-2.5-pro", "screenshot", 100, ["self_reported"], "s2_5/results_gemini_2p5_uitars_72b_100_steps.zip (grounding UI-TARS-72B)", ""),
    "19P45SaQGgktuGSv_zWxzYRkqSx0aiJSk": ("agent s2 w/ gemini-2.5-pro", "screenshot", 50, ["self_reported"], "s2_5/results_gemini_2p5_uitars_72b_50_steps.zip; differs from the HF 50-step run on 18 tasks", ""),
    "1aCzGfmBUU_CM6O0xMeOnlh_xRiTA5tjN": ("agent s2 w/ gemini-2.5-pro-preview-03-25", None, 15, ["self_reported"], "s2_and_before/results_gemini_2_5_pro_15_step.zip", ""),
    "15DmqvnXDKNbinU3QICfYRahHCMyhbQma": ("agent s2 w/ gemini-2.5-pro-preview-03-25", None, 50, ["self_reported"], "s2_and_before/results_gemini_2_5_pro_50_step.zip", ""),
    "1oR2e0QHhzYJh5R-m22SK8orypRyPXz9n": ("agent s2 w/ gpt-4o", None, 15, ["self_reported"], "s2_and_before/results_gpt4o_15_step_wo_infeasible.zip", "variant=wo_infeasible"),
    "1y4PG8HeFR6J1vOqi3peQqq7xHuLfgPGN": ("agent s2 w/ gpt-4o", None, 50, ["self_reported"], "s2_and_before/results_gpt4o_50_step_wo_infeasible.zip", "variant=wo_infeasible"),
    "1eS-bFBpYkLLhK7AUQIfv_m4h6NM2Cvkp": ("agent s2 w/ claude-3-5-sonnet-20241022", None, 15, ["self_reported"], "s2_and_before/results_sonnet_3_5_15_step_wo_infeasible.zip", "variant=wo_infeasible"),
    "1r5mhSrhO_86XN4jpmdbWFdFC3k9GdZga": ("agent s2 w/ claude-3-5-sonnet-20241022", None, 50, ["self_reported"], "s2_and_before/results_sonnet_3_5_50_step_with_infeasible.zip", "variant=with_infeasible"),
    "1ZLo7ytn8xEPqzb3FLNhfsOMnc6OoqKHF": ("agent s2 w/ claude-3-5-sonnet-20241022", None, 50, ["self_reported"], "s2_and_before/results_sonnet_3_5_50_step_wo_infeasible.zip", "variant=wo_infeasible"),
    "19RppvbiJ2Ajdfy9vBICw3JWeQHJwztzd": ("agent s2 w/ claude-3-7-sonnet-20250219 (thinking)", None, 15, ["self_reported"], "s2_and_before/results_sonnet_3_7_thinking_15_step_with_infeasible.zip", "variant=with_infeasible"),
    "1TPfH3wY0ujBHioZHRztwIFl45mOvBIbg": ("agent s2 w/ claude-3-7-sonnet-20250219 (thinking)", None, 15, ["self_reported"], "s2_and_before/results_sonnet_3_7_thinking_15_step_wo_infeasible.zip", "variant=wo_infeasible"),
    "1A6UY39J-6MM2S5BghMLsFUIjr4U3mvhK": ("agent s2 w/ claude-3-7-sonnet-20250219 (thinking)", None, 50, ["self_reported"], "s2_and_before/results_sonnet_3_7_thinking_50_step_with_infeasible.zip", "variant=with_infeasible"),
    "1BbtYapsRdcwdnHLpYBtxIO6ukT185Rc4": ("agent s2 w/ claude-3-7-sonnet-20250219 (thinking)", None, 50, ["self_reported"], "s2_and_before/results_sonnet_3_7_thinking_50_step_wo_infeasible.zip", "variant=wo_infeasible"),
}
# Drive "last modified" dates (folder listing), used only when a zip carries no file timestamps
DRIVE_UPLOAD_DATE = {"1BbtYapsRdcwdnHLpYBtxIO6ukT185Rc4": "2025-04-03"}
# all_result.json only (folder-style uploads); possible "False"->1.0 counting bug cannot be undone
DRIVE_ALLRESULT = {
    "1p83vSc48O1cB8ld2pWai35lLR4dHMhRB": ("GPT-4 Vision (0409) (gpt-4-turbo-2024-04-09)", "screenshot", 15, "folder gpt-4-turbo-2024-04-09/all_result.json"),
    "1LXe4zhTsQ-_HQl9it2nOlLAirqnSu633": ("Gemini-Pro-1.5", "screenshot", 15, "folder gemini-1.5-pro-latest/all_result.json"),
    "1VVwKu-QT26GjyR--emUwenXqZ1jNC4Mc": ("Qwen-vl-Max-0809", "screenshot", 15, "folder qwen-vl-max/all_result.json"),
}
RMCTS_ID = "1F5R0NmEeHc-VysE0my_dzwA2b2-xHa1M"

# ----------------------------------------------------------------------------- collect runs
runs = []      # dicts with run_id, model, config, flags, scores{(dom,tid): score}, run_date, ...
problems = []


def hf_members(z):
    return json.load(open(os.path.join(DL, "members", z, "members.json")))


def hf_listing_dates(z):
    p = os.path.join(DL, "listings", z + ".json")
    if not os.path.exists(p):
        return {}
    return {m["name"]: m["date"] for m in json.load(open(p))["members"]}


def tar_dates(z):
    p = os.path.join(DL, "members", z, "tar_index.json")
    if not os.path.exists(p):
        return {}
    return {n: list(datetime.datetime.fromtimestamp(m, datetime.timezone.utc).timetuple()[:6]) for n, s, m in json.load(open(p))}


def args_of(M, root):
    for n, v in M.items():
        if n.endswith("args.json") and (root == "" or n.startswith(root) or root in n):
            try:
                return json.loads(v)
            except Exception:
                pass
    for n, v in M.items():
        if n.endswith("args.json"):
            try:
                return json.loads(v)
            except Exception:
                pass
    return {}


seen_containers = set()
for z, roots, row, extra in HF:
    seen_containers.add(z)
    M = hf_members(z)
    dates = hf_listing_dates(z) or tar_dates(z)
    scores, sdates, bad, conflict = {}, [], 0, 0
    if extra.get("score_file") == "all_result.json":
        ar = [v for n, v in M.items() if n.endswith("all_result.json")][0]
        ar = load_json_any(ar)
        rt = {}
        for n, v in M.items():
            t = task_of(n)
            if t and n.endswith("/result.txt"):
                rt[(t[0], t[1])] = parse_score(v)
                if dates.get(n):
                    sdates.append("%04d-%02d-%02d" % tuple(dates[n][:3]))
        for d, dd in ar.items():
            for tid, v in dd.items():
                if rt.get((d, tid)) is not None and abs(rt[(d, tid)] - float(v)) > 1e-9:
                    conflict += 1
                    continue
                scores[(d, tid)] = float(v)
    else:
        for n, v in M.items():
            t = task_of(n)
            if not t or os.path.basename(n) not in ("result.txt",):
                continue
            d, tid, root = t
            if not any((r == "" or r in root) for r in roots):
                continue
            if len(roots) > 1 or roots != [""]:
                pass
            s = parse_score(v)
            if s is None:
                bad += 1
                continue
            key = (d, tid)
            if key in scores:   # repair run overrides the failed main attempt (Holo3 run1 + repair)
                pass
            scores[key] = s
            if dates.get(n):
                sdates.append("%04d-%02d-%02d" % tuple(dates[n][:3]))
    # sanity: one root per catalogue entry unless listed
    rootset = set()
    for n in M:
        t = task_of(n)
        if t and n.endswith("/result.txt") and any((r == "" or r in t[2]) for r in roots):
            rootset.add(t[2])
    if roots == [""] and len(rootset) > 1 and not extra.get("score_file"):
        raise SystemExit(f"{z}: several roots {rootset}")
    a = args_of(M, list(rootset)[0] if rootset else "")
    L = LB.get(row, {}) if row is not None else {}
    model = extra.get("model") or L.get("model")
    ms = L.get("max_steps") if row is not None else extra.get("max_steps")
    if a.get("max_steps") and ms and int(a["max_steps"]) != int(ms):
        extra.setdefault("note", "")
        extra["note"] += f"; args.max_steps={a['max_steps']} differs from leaderboard {ms}"
    runs.append(dict(
        run_id=f"hf:{z}" + (f"#{roots[0]}" if roots != [""] else ""), source="hf_verified", container=z,
        roots=";".join(sorted(rootset)), model=model, institution=L.get("institution", extra.get("institution")),
        approach=L.get("approach", extra.get("approach")), max_steps=ms,
        obs=a.get("observation_type", "screenshot"), action_space=a.get("action_space"),
        a11y=L.get("a11y"), coding=L.get("coding"), multi=L.get("multi"), args_model=a.get("model"),
        lb_row=row, lb_model=L.get("model"), lb_date=L.get("date"), lb_sum=L.get("lb_sum"), lb_n=L.get("lb_n"),
        version="OSWorld-Verified", flags=list(extra.get("flags", [])), note=extra.get("note", ""),
        scores=scores, run_date=min(sdates) if sdates else L.get("date"), n_unparsed=bad, n_conflict_dropped=conflict))

# Holo3 internal fallback runs (nested zips; task_summary.json reward)
for t in HOLO_INTERNAL:
    M = json.load(open(os.path.join(DL, "members", f"holo3_internal_{t}", "members.json")))
    dates = {m["name"]: m["date"] for m in json.load(open(os.path.join(DL, "listings", f"holo3_internal_{t}.json")))["members"]}
    scores, sdates = {}, []
    for n, v in M.items():
        tt = task_of(n)
        if tt and n.endswith("task_summary.json"):
            r = json.loads(v).get("reward")
            if r is None or not (0 <= float(r) <= 1):
                continue
            scores[(tt[0], tt[1])] = float(r)
            sdates.append("%04d-%02d-%02d" % tuple(dates[n][:3]))
    runs.append(dict(
        run_id=f"hf:holo3_internal_{t}", source="hf_verified_submitter_internal",
        container="OSWorldSurfer-Holo3-35B-A3B-20260330-OSWorld-Verified_hcompany-internal-runs-20260416_complete-with-rewards.zip"
                  f"/h_company_internal_runs/trajectories_results_20260416_{t}.zip",
        roots=f"trajectories/results_20260416_{t}", model="Holo3-35B-A3B", institution="H Company", approach="Specialized model",
        max_steps=None, obs="screenshot", action_space=None, a11y=None, coding=None, multi=None, args_model=None,
        lb_row=None, lb_model=None, lb_date=None, lb_sum=None, lb_n=None, version="OSWorld-Verified",
        flags=["self_reported", "not_on_leaderboard"], note="H Company internal fallback run (agent OSWorldSurfer); reward from task_summary.json",
        scores=scores, run_date=min(sdates), n_unparsed=0, n_conflict_dropped=0))

# Drive runs (original OSWorld, 2024-2025)
DD = os.path.join(DL, "drive")
for i, tup in DRIVE.items():
    model, setting, ms, flags, note = tup[:5]
    variant = tup[5] if len(tup) > 5 else ""
    mp = os.path.join(DD, "members", i, "members.json")
    if not os.path.exists(mp):
        print("missing drive run", i, model, setting)
        continue
    M = json.load(open(mp))
    dates = {m["name"]: m["date"] for m in json.load(open(os.path.join(DD, "listings", i + ".json")))["members"]}
    scores, sdates, bad, roots = {}, [], 0, set()
    for n, v in M.items():
        t = task_of(n)
        if not t or os.path.basename(n) != "result.txt":
            continue
        s = parse_score(v)
        roots.add(t[2])
        if s is None:
            bad += 1
            continue
        scores[(t[0], t[1])] = s
        if dates[n][0] >= 2020:                       # some zips carry no timestamps (1980-01-01)
            sdates.append("%04d-%02d-%02d" % tuple(dates[n][:3]))
    if len(roots) > 1:
        raise SystemExit(f"drive {i}: several roots {roots}")
    a = args_of(M, "")
    root = ";".join(roots)
    mobs = re.search(r"(?:^|/)(pyautogui|computer_13)/([a-z_0-9]+)/", root + "/")
    obs = setting or (a.get("observation_type") or (mobs.group(2) if mobs else "screenshot"))
    runs.append(dict(
        run_id=f"drive:{i}", source="drive_original", container=f"drive file {i}", roots=root, model=model, variant=variant,
        institution=None, approach=None, max_steps=a.get("max_steps", ms), obs=obs, action_space=a.get("action_space", "pyautogui"),
        a11y=None, coding=None, multi=None, args_model=a.get("model"), lb_row=None, lb_model=None, lb_date=None, lb_sum=None, lb_n=None,
        version="OSWorld-original", flags=["pre_verified_version"] + list(flags), note=note, scores=scores,
        run_date=min(sdates) if sdates else DRIVE_UPLOAD_DATE.get(i), n_unparsed=bad, n_conflict_dropped=0))
    if runs[-1]["run_date"] and runs[-1]["run_date"] >= VERIFIED_LAUNCH:   # e.g. Agent S2.5 uploads of Aug 2025
        runs[-1]["version"] = "OSWorld-Verified"
        runs[-1]["flags"] = [f for f in runs[-1]["flags"] if f != "pre_verified_version"]

for i, (model, setting, ms, note) in DRIVE_ALLRESULT.items():
    ar = load_json_any(open(os.path.join(DD, "small", i + ".bin")).read())
    scores = {(d, t): float(v) for d, dd in ar.items() for t, v in dd.items()}
    runs.append(dict(
        run_id=f"drive:{i}", source="drive_original_all_result_only", container=f"drive file {i}", roots=note, model=model,
        institution=None, approach=None, max_steps=ms, obs=setting, action_space="pyautogui", a11y=None, coding=None, multi=None,
        args_model=None, lb_row=None, lb_model=None, lb_date=None, lb_sum=None, lb_n=None, version="OSWorld-original",
        flags=["pre_verified_version", "all_result_only"], note=note + "; per-task values from all_result.json only (see README caveat)",
        scores=scores, run_date={"1p83vSc48O1cB8ld2pWai35lLR4dHMhRB": "2024-04-23", "1LXe4zhTsQ-_HQl9it2nOlLAirqnSu633": "2024-05-02",
                                 "1VVwKu-QT26GjyR--emUwenXqZ1jNC4Mc": "2024-05-23"}[i],
        n_unparsed=0, n_conflict_dropped=0))

# R-MCTS (tar.gz from Drive, read in memory)
raw = gzip.decompress(open(os.path.join(DD, "small", RMCTS_ID + ".bin"), "rb").read())
scores, sdates, bad = {}, [], 0
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
    for ti in tf:
        t = task_of(ti.name)
        if t and ti.isfile() and ti.name.endswith("/result.txt"):
            s = parse_score(tf.extractfile(ti).read().decode("utf-8", "replace"))
            if s is None:
                bad += 1
                continue
            scores[(t[0], t[1])] = s
            sdates.append(datetime.datetime.fromtimestamp(ti.mtime, datetime.timezone.utc).date().isoformat())
runs.append(dict(
    run_id=f"drive:{RMCTS_ID}", source="drive_original", container=f"drive file {RMCTS_ID} (rmcts_screenshot-a11ytree_osworld_submission.gz)",
    roots="v0.4.1_submission_screenshot_a11ytree", model="R-MCTS w/ GPT-4o (max 60 tree nodes)", institution="Columbia University & Microsoft",
    approach="Agentic framework", max_steps=None, obs="screenshot_a11y_tree", action_space=None, a11y=None, coding=None, multi=None,
    args_model=None, lb_row=None, lb_model=None, lb_date=None, lb_sum=None, lb_n=None, version="OSWorld-original",
    flags=["pre_verified_version", "self_reported", "custom_harness"], note="tree search agent; result.txt per task",
    scores=scores, run_date=min(sdates), n_unparsed=bad, n_conflict_dropped=0))

# ----------------------------------------------------------------------------- instruction versions
def git(*a, inp=None):
    return subprocess.run(["git", "--git-dir", REPO, *a], capture_output=True, input=inp, check=True).stdout


snap_cache = {}


def snapshot(date):
    if date == "HEAD":
        c = HEAD_SHA
    else:
        c = git("rev-list", "--first-parent", "-1", f"--before={date}T00:00:00+00:00", HEAD_SHA).decode().strip()
    if c in snap_cache:
        return snap_cache[c]
    cd = git("show", "-s", "--format=%cI", c).decode().strip()
    paths = {}
    for ln in git("ls-tree", "-r", c, "evaluation_examples/examples").decode().split("\n"):
        if ln.strip() and ln.endswith(".json"):
            meta, path = ln.split("\t", 1)
            paths[path] = meta.split()[2]
    shas = sorted(set(paths.values()))
    rawb = git("cat-file", "--batch", inp=("\n".join(shas) + "\n").encode())
    content, k = {}, 0
    while k < len(rawb):
        nl = rawb.index(b"\n", k)
        sha, typ, size = rawb[k:nl].decode().split()
        size = int(size)
        content[sha] = rawb[nl + 1: nl + 1 + size]
        k = nl + 1 + size + 1
    tasks = {}
    for p, s in paths.items():
        try:
            j = json.loads(content[s])
        except Exception:
            continue
        tasks[(p.split("/")[-2], p.split("/")[-1][:-5])] = dict(instruction=j.get("instruction"), source=j.get("source"),
                                                                related_apps=j.get("related_apps"))
    lists = {}
    for lf in ("test_all.json", "test_nogdrive.json"):
        try:
            lists[lf] = json.loads(git("show", f"{c}:evaluation_examples/{lf}"))
        except Exception:
            lists[lf] = None
    snap_cache[c] = dict(commit=c, commit_date=cd, tasks=tasks, lists=lists)
    return snap_cache[c]


head = snapshot("HEAD")
neurips = snapshot("2024-12-10")          # state of main during NeurIPS 2024 (Dec 10-15, 2024)
assert head["commit"] == HEAD_SHA

versions = collections.defaultdict(list)   # (dom,tid) -> list of dict(text, first_date, snaps)


def version_index(key, text, date, commit):
    V = versions[key]
    for k, v in enumerate(V):
        if v["text"] == text:
            v["dates"].add(date); v["commits"].add(commit)
            return k
    V.append(dict(text=text, dates={date}, commits={commit}))
    return len(V) - 1


rows, missing_text = [], collections.Counter()
bad_dates = [(r["run_id"], r["run_date"]) for r in runs if not r["run_date"] or not re.match(r"^20\d\d-\d\d-\d\d$", r["run_date"])]
if bad_dates:
    raise SystemExit(f"runs without a usable date: {bad_dates}")
for r in runs:
    snap = snapshot(r["run_date"])
    r["snapshot_commit"], r["snapshot_date"] = snap["commit"], snap["commit_date"]
    for (d, tid), s in r["scores"].items():
        info = snap["tasks"].get((d, tid))
        src = "run_date_snapshot"
        if info is None or not info.get("instruction"):
            # task file absent on main at run date: use the nearest snapshot that has it (HEAD, then NeurIPS 2024)
            for alt in (head, neurips):
                if alt["tasks"].get((d, tid), {}).get("instruction"):
                    info, src = alt["tasks"][(d, tid)], "fallback_" + ("head" if alt is head else "neurips2024")
                    break
        if info is None:
            missing_text[(d, tid)] += 1
            continue
        vi = version_index((d, tid), info["instruction"], snap["commit_date"][:10], snap["commit"])
        rows.append(dict(run_id=r["run_id"], dom=d, tid=tid, vi=vi, score=s, text_source=src))

# make sure HEAD and NeurIPS-2024 versions of every test_all task exist as items
for snapx, tag in ((head, "head"), (neurips, "neurips2024")):
    for d, ids in (snapx["lists"]["test_all.json"] or {}).items():
        for tid in ids:
            info = snapx["tasks"].get((d, tid))
            if info and info.get("instruction"):
                version_index((d, tid), info["instruction"], snapx["commit_date"][:10], snapx["commit"])

# order versions by first date and build item keys
item_rows, vmap = [], {}
head_all = {(d, t) for d, ids in head["lists"]["test_all.json"].items() for t in ids}
head_nog = {(d, t) for d, ids in head["lists"]["test_nogdrive.json"].items() for t in ids}
neu_all = {(d, t) for d, ids in neurips["lists"]["test_all.json"].items() for t in ids}
for key, V in versions.items():
    order = sorted(range(len(V)), key=lambda k: min(V[k]["dates"]))
    for newk, k in enumerate(order, 1):
        vmap[(key, k)] = newk
        d, tid = key
        txt = V[k]["text"]
        is_head = head["tasks"].get(key, {}).get("instruction") == txt
        is_neu = neurips["tasks"].get(key, {}).get("instruction") == txt
        hinfo = head["tasks"].get(key) or neurips["tasks"].get(key) or {}
        item_rows.append(dict(item_key=f"{d}/{tid}@v{newk}", text=txt, extra=json.dumps(dict(
            domain=d, task_id=tid, version=newk, n_versions=len(V),
            snapshot_dates=sorted(V[k]["dates"]), is_head_version=is_head, is_neurips2024_version=is_neu,
            in_test_all_head=key in head_all, in_test_nogdrive_head=key in head_nog, in_test_all_neurips2024=key in neu_all,
            source=hinfo.get("source"), related_apps=hinfo.get("related_apps"),
            task_file=f"evaluation_examples/examples/{d}/{tid}.json"), ensure_ascii=False)))
items = pd.DataFrame(item_rows).sort_values("item_key").reset_index(drop=True)
# link items whose instruction text is identical (e.g. the 15 Thunderbird tasks renamed on 2025-02-10
# because their 2024 ids duplicated Chrome task ids)
by_text = items.groupby("text").item_key.apply(list).to_dict()
ex = []
for k, t, e in zip(items.item_key, items.text, items.extra):
    e = json.loads(e)
    e["same_text_item_keys"] = [x for x in by_text[t] if x != k]
    ex.append(json.dumps(e, ensure_ascii=False))
items["extra"] = ex
assert items.item_key.is_unique

# ----------------------------------------------------------------------------- results
R = {r["run_id"]: r for r in runs}


def config_of(r):
    parts = [r["version"]]
    if r["approach"]:
        parts.append(f"approach={r['approach']}")
    parts.append(f"max_steps={r['max_steps'] if r['max_steps'] is not None else 'unknown'}")
    parts.append(f"obs={r['obs']}")
    if r["action_space"]:
        parts.append(f"action_space={r['action_space']}")
    for k, lab in (("a11y", "extra_a11y_tree"), ("coding", "coding_action"), ("multi", "multiple_rollout")):
        if r[k] is not None and not (isinstance(r[k], float) and pd.isna(r[k])):
            parts.append(f"{lab}={r[k]}")
    if r.get("variant"):
        parts.append(r["variant"])
    if r["source"] in ("drive_original", "drive_original_all_result_only") and "self_reported" not in r["flags"] and r["approach"] is None:
        parts.append("baseline_agent=OSWorld PromptAgent (t=1.0, top_p=0.9)")
    return " | ".join(parts)


res = []
for x in rows:
    r = R[x["run_id"]]
    res.append(dict(item_key=f"{x['dom']}/{x['tid']}@v{vmap[((x['dom'], x['tid']), x['vi'])]}", model=r["model"], config=config_of(r),
                    run_id=r["run_id"], outcome=float(x["score"]), score_is_fractional=not float(x["score"]).is_integer(),
                    text_source=x["text_source"], flags=",".join(r["flags"])))
results = pd.DataFrame(res)
results = results.sort_values(["model", "config", "run_id", "item_key"]).reset_index(drop=True)
# trial = 1-based index within (model, config, task) ordered by run date then run id
results["task"] = results.item_key.str.split("@").str[0]
run_order = {r["run_id"]: (r["run_date"] or "", r["run_id"]) for r in runs}
results["_o"] = results.run_id.map(run_order)
results = results.sort_values(["model", "config", "task", "_o"]).reset_index(drop=True)
results["trial"] = results.groupby(["model", "config", "task"]).cumcount() + 1
results = results.drop(columns=["_o", "task"])
results = results[["item_key", "model", "config", "trial", "outcome", "run_id", "flags", "score_is_fractional", "text_source"]]
assert not results.duplicated(["item_key", "model", "config", "trial"]).any()

# ----------------------------------------------------------------------------- run table + checks
run_rows = []
for r in runs:
    vals = list(r["scores"].values())
    kept = results[results.run_id == r["run_id"]]
    run_rows.append(dict(
        run_id=r["run_id"], source=r["source"], model=r["model"], config=config_of(r), flags=",".join(r["flags"]),
        container=r["container"], roots=r["roots"], args_model=r["args_model"], run_date=r["run_date"],
        task_snapshot_commit=r["snapshot_commit"], task_snapshot_date=r["snapshot_date"],
        n_graded=len(vals), n_rows=len(kept), score_sum=round(sum(vals), 4), success_rate=round(sum(vals) / len(vals), 4) if vals else None,
        n_unparsed_result_files=r["n_unparsed"], n_conflict_dropped=r["n_conflict_dropped"],
        lb_row=r["lb_row"], lb_model=r["lb_model"], lb_date=r["lb_date"], lb_sum=r["lb_sum"], lb_n=r["lb_n"],
        lb_match=(None if r["lb_sum"] is None else (abs(r["lb_sum"] - sum(vals)) < 0.015 and r["lb_n"] == len(vals))),
        note=r["note"]))
runs_df = pd.DataFrame(run_rows)

# ----------------------------------------------------------------------------- models table
REL = {  # model -> (release date or month, basis)
    "o3": ("2025-04-16", "OpenAI o3 release"),
    "claude-3-7-sonnet-20250219": ("2025-02-24", "model id"), "claude-4-sonnet-20250514": ("2025-05-22", "model id"),
    "claude-sonnet-4-5-20250929": ("2025-09-29", "model id"), "claude-sonnet-4-6": ("2026-02", "leaderboard row 2026; Anthropic, '26"),
    "doubao-1-5-thinking-vision-pro-250428": ("2025-04-28", "model id"), "doubao-1-5-thinking-vision-pro-250717": ("2025-07-17", "model id"),
    "uitars-1.5-7b": ("2025-04-16", "UI-TARS-1.5 release"), "uitars-72b-dpo": ("2025-01-21", "UI-TARS release"),
    "opencua-a3b": ("2025-08", "OpenCUA paper (Wang et al. '25)"), "opencua-qwen2-7b": ("2025-08", "OpenCUA paper"),
    "opencua-7b": ("2025-08", "OpenCUA paper"), "opencua-32b": ("2025-08", "OpenCUA paper"),
    "qwen2.5-vl-32b-instruct": ("2025-03-24", "Qwen2.5-VL-32B release"), "qwen2.5-vl-72b-instruct": ("2025-01-28", "Qwen2.5-VL release"),
    "kimi-vl-a3b": ("2025-04-10", "Kimi-VL release"),
    "agent s2 w/ gemini-2.5-pro": ("2025-04", "Agent S2 w/ Gemini 2.5 (self-reported 2025-04-18)"),
    "agent s2.5 w/ o3": ("2025-07", "leaderboard 2025-07-31"),
    "Jedi-7B w/ gpt-4o": ("2025-05", "Jedi paper (Xie et al. '25)"), "Jedi-7B w/ o3": ("2025-05", "Jedi paper"),
    "GTA1 w/ o3": ("2025-07", "GTA1 paper (Yang et al. '25)"), "CoACT-1": ("2025-08", "CoAct-1 paper (Song et al. '25)"),
    "TianXi-Action-7B": ("2025-06", "Drive upload 2025-06-18; leaderboard 2025-08-11"),
    "autoglm-os-9b": ("2025-08", "leaderboard 2025-08-18"), "autoglm-os-9b-20250925": ("2025-09-25", "model id"),
    "gui-owl-7b": ("2025-08", "Mobile-Agent-v3 / GUI-Owl paper"), "mobile-agent-v3 w/ gui-owl-32b": ("2025-08", "Mobile-Agent-v3 paper"),
    "Agentic-Lybic-Maestro": ("2025-09", "leaderboard 2025-09-09"), "UI-TARS-250705": ("2025-07-05", "model id"),
    "UiPath Screen Agent w/ GPT-5": ("2025-09", "leaderboard 2025-09-14"), "DeepMiner-Mano-7B": ("2025-09", "leaderboard 2025-09-18"),
    "agent s3 w/ Opus 4.5 + GPT-5 bBoN (N=10)": ("2025-12", "leaderboard 2025-12-11"), "agent s3 w/ Opus 4.5 bBoN (N=1)": ("2025-12", "leaderboard 2025-12-11"),
    "UI-TARS-2-2509": ("2025-09", "model id"), "GBOX Agent": ("2025-11", "leaderboard 2025-11-25"), "Seed-1.8": ("2025-12", "leaderboard 2025-12-18"),
    "EvoCUA-20260105": ("2026-01-05", "model id"), "EvoCUA-8B-20260105": ("2026-01-05", "model id"),
    "DART-GUI-7B-0924": ("2025-09-24", "model id"), "Kimi K2.5": ("2026-01", "leaderboard 2026-01-31"),
    "HIPPO Agent w/ Opus 4.5": ("2026-02", "leaderboard 2026-02-25"), "Kimi K2.6": ("2026-04", "leaderboard 2026-04-20"),
    "OpenAPA w/ gemini-3.1-pro": ("2026-04", "leaderboard 2026-04-17"), "Holo3-35B-A3B": ("2026-03-30", "agent id 20260330"),
    "VLAA-GUI w/ Opus 4.5": ("2026-04", "leaderboard 2026-04-20"), "Qwen 3.7 Plus": ("2026-05", "leaderboard 2026-05-25"),
    "MiniMax M3": ("2026-06", "leaderboard 2026-06-08"), "Muse Spark 1.1": ("2026-07", "leaderboard 2026-07-09"),
    "Intelligence-Indeed Agent": ("2026-07", "leaderboard 2026-07-25"), "claude-fable-5[1m]": ("2026-07", "leaderboard 2026-08-01; run 2026-07-25"),
    "claude-opus-5[1m]": ("2026-07", "leaderboard 2026-08-01; run 2026-07-26"),
    "Qwen3.5-Plus (internal checkpoint qwen35_plus_nothink_cpt_gemini_s1_0213)": ("2026-02", "run date 2026-02-14 (checkpoint 0213)"),
    "AGI-0": ("2025-10", "run dates 2025-10-17 and 2025-11-17"),
    "GPT-4 Vision (gpt-4-vision-preview)": ("2023-11-06", "OpenAI DevDay"), "Gemini-Pro Vision": ("2023-12-13", "Gemini Pro API"),
    "GPT-4o": ("2024-05-13", "OpenAI"), "GPT-4o-mini": ("2024-07-18", "OpenAI"), "InternVL2": ("2024-07", "OpenGVLab"),
    "Llava-OneVision": ("2024-08", "LLaVA-OneVision release"), "MiniCPM-V 2.6": ("2024-08", "OpenBMB"),
    "GPT-4 (gpt-4-0125-preview)": ("2024-01-25", "model id"), "GPT-4 Vision (0409) (gpt-4-turbo-2024-04-09)": ("2024-04-09", "model id"),
    "Qwen-Max": ("2024", "self-reported table 2024-05-22"), "Gemini-Pro-1.5": ("2024-02", "Gemini 1.5 Pro"),
    "GPT-3.5 (gpt-3.5-turbo-16k)": ("2023-06", "OpenAI"), "Gemini-Pro": ("2023-12-13", "Gemini Pro API"),
    "Llama-3-70B": ("2024-04-18", "Meta"), "Qwen-vl-Max-0809": ("2024-08-09", "model id"),
    "Claude-3-Opus (claude-3-opus-20240229)": ("2024-02-29", "model id"),
    "InfantAgent (Claude-3.7 Sonnet planner + UI-TARS-1.5-7B grounder)": ("2025-04", "self-reported 2025-04-30"),
    "Jedi-3B w/ gpt-4o": ("2025-05", "Jedi paper"), "R-MCTS w/ GPT-4o (max 60 tree nodes)": ("2024-12", "self-reported 2024-12-20 / 2025-01-14"),
    "agent s2 w/ gpt-4o": ("2025-03", "Agent S2 (2025-03); GPT-4o 2024-05"),
    "agent s2 w/ claude-3-5-sonnet-20241022": ("2025-03", "Agent S2 (2025-03); Claude 3.5 Sonnet 2024-10"),
    "agent s2 w/ claude-3-7-sonnet-20250219 (thinking)": ("2025-03", "Agent S2 w/ Claude 3.7 (2025-03-13)"),
    "agent s2 w/ gemini-2.5-pro-preview-03-25": ("2025-04", "Agent S2 w/ Gemini 2.5 (2025-04-18)"),
}
mrows = []
for m, g in results.groupby("model"):
    rd, basis = REL.get(m, (None, "unknown"))
    yr = int(rd[:4]) if rd else None
    mrows.append(dict(model=m, release_date=rd, release_year=yr, basis=basis, is_2025_2026=(yr in (2025, 2026)) if yr else None,
                      n_results=len(g), n_runs=g.run_id.nunique(), n_configs=g.config.nunique()))
models_df = pd.DataFrame(mrows).sort_values(["release_date", "model"], na_position="last")
unknown = models_df[models_df.release_year.isna()]
if len(unknown):
    print("models without release date:", list(unknown.model))

# ----------------------------------------------------------------------------- write
os.makedirs(OUT, exist_ok=True)
items.to_parquet(os.path.join(OUT, "items.parquet"), index=False)
results.to_parquet(os.path.join(OUT, "results.parquet"), index=False)
runs_df.to_csv(os.path.join(OUT, "runs.csv"), index=False)
models_df.to_csv(os.path.join(OUT, "models.csv"), index=False)
summary = dict(
    n_items=len(items), n_tasks=items.item_key.str.split("@").str[0].nunique(), n_results=len(results),
    n_models=results.model.nunique(), n_runs=results.run_id.nunique(), n_configs=results.groupby(["model", "config"]).ngroups,
    n_models_2025_2026=int(models_df.is_2025_2026.fillna(False).sum()),
    n_items_with_results=results.item_key.nunique(), n_fractional=int(results.score_is_fractional.sum()),
    text_source=results.text_source.value_counts().to_dict(), missing_text=dict((f"{a}/{b}", c) for (a, b), c in missing_text.items()),
    by_source=runs_df.groupby("source").agg(runs=("run_id", "count"), rows=("n_rows", "sum")).to_dict("index"),
    lb_mismatch=runs_df[runs_df.lb_match == False][["run_id", "score_sum", "n_graded", "lb_sum", "lb_n"]].to_dict("records"),
    head_commit=head["commit"], neurips_snapshot=neurips["commit"], neurips_snapshot_date=neurips["commit_date"],
    versions_per_task=collections.Counter(len(v) for v in versions.values()),
)
print(json.dumps(summary, indent=1, default=str))
json.dump(summary, open(os.path.join(OUT, "build_summary.json"), "w"), indent=1, default=str)
