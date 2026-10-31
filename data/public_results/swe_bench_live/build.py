"""Build normalized SWE-bench-Live per-item results.

Outputs items.parquet, results.parquet and submissions.csv for three split families:
  lite      -> HF SWE-bench-Live/SWE-bench-Live (all splits), submissions/lite/
  multilang -> HF SWE-bench-Live/MultiLang (c, cpp, cs, go, java, js, rust, ts), submissions/multilang/
  windows   -> HF SWE-bench-Live/Windows (test), submissions/windows/

Sources (pinned revisions):
  https://github.com/SWE-bench-Live/submission @ 641b727c8a5e3e5d0af08a348e632b12c3f11aef (submissions/)
  https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live @ b51a86422e10cfd403beb4773e5a2947953e36ec
  https://huggingface.co/datasets/SWE-bench-Live/MultiLang @ 3638632e8153a10ca422c1022bed79023084b5c9
  https://huggingface.co/datasets/SWE-bench-Live/Windows @ b9fa25f2732dbc2c93b9738b610d053d8b7bd7a9
Outcome: from each submission's results.json (result.json in multilang/all_languages): resolved_ids / success_ids -> 1.0,
unresolved_ids / failure_ids -> 0.0. IDs only in error_ids, incomplete_ids or empty_patch_ids get no row. Dropped: graded IDs
also listed as error/incomplete, resolved IDs also listed as empty patch, and IDs not in the pinned HF revision.

Usage: python -I build.py <submission_clone> <hf_main_dir> <hf_multilang_dir> <hf_windows_dir> <out_dir> <log.json>
"""
import json, glob, os, sys
import pandas as pd

SUB_ROOT, HF_MAIN, HF_ML, HF_WIN, OUT, LOG = sys.argv[1:7]

FAMILIES = {
    # family: (HF dataset, local snapshot dir, split order)
    'lite': ('SWE-bench-Live/SWE-bench-Live', HF_MAIN, ['lite', 'verified', 'test', 'full']),
    'multilang': ('SWE-bench-Live/MultiLang', HF_ML, ['c', 'cpp', 'cs', 'go', 'java', 'js', 'rust', 'ts']),
    'windows': ('SWE-bench-Live/Windows', HF_WIN, ['test']),
}

# ---------------- items ----------------
items, split_stats = [], {}
for fam, (dataset, root, splits) in FAMILIES.items():
    parts = []
    for s in splits:
        files = sorted(glob.glob(f'{root}/data/{s}-*.parquet'))
        assert files, (fam, s)
        for f in files:
            d = pd.read_parquet(f)
            keep = ['instance_id', 'repo', 'pull_number', 'base_commit', 'created_at', 'problem_statement'] + (['language'] if 'language' in d.columns else [])
            d = d[keep].copy()
            d['split'] = s
            parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    # one text / base_commit / repo / created_at per instance_id across all splits and duplicate rows
    nun = raw.groupby('instance_id')[['problem_statement', 'base_commit', 'repo', 'created_at']].nunique()
    assert (nun <= 1).all().all(), f'{fam}: conflicting fields for {(nun > 1).any(axis=1).sum()} ids'
    split_stats[fam] = {s: [int((raw.split == s).sum()), int(raw.loc[raw.split == s, 'instance_id'].nunique())] for s in splits}
    for iid, g in raw.groupby('instance_id', sort=True):
        r = g.iloc[0]
        ca = r.created_at
        extra = {
            'dataset': dataset,
            'split_family': fam,
            'repo': r.repo,
            'pull_number': r.pull_number,
            'base_commit': r.base_commit,
            # lite: parquet timestamp (UTC, naive) -> ISO; multilang/windows: ISO string exactly as released
            'created_at': pd.Timestamp(ca).isoformat() if not isinstance(ca, str) else ca,
            'splits': [s for s in splits if s in set(g.split)],
        }
        if 'language' in g.columns:
            extra['language'] = r.language
        items.append({'item_key': iid, 'text': r.problem_statement, 'extra': json.dumps(extra, ensure_ascii=False), '_fam': fam})
items = pd.DataFrame(items)
assert items.item_key.is_unique, 'instance_id collides across split families'
fam_ids = {fam: set(items.loc[items._fam == fam, 'item_key']) for fam in FAMILIES}
items = items.drop(columns='_fam')

# ---------------- submissions ----------------
# path under submissions/ -> (model as given in the submission, config = agent/scaffold [+ version/setting], flags)
# flags: success_only = package lists only resolved ids; custom_harness = graded by the submitter's own harness,
#        not the official SWE-bench-Live evaluator; retry_on_failure = README says failed instances were re-run.
SUBS = {
    # ---- lite (Python) ----
    'lite/20250725-openhands-Qwen3-Coder-480B-A35B': ('Qwen3-Coder-480B-A35B', 'OpenHands', []),
    'lite/20251221-MITIBM-agent-seedoss36b': ('Seed-OSS-36B-Instruct', 'MIT-IBM Agent + SWE-agent', []),
    'lite/Slai-Agent/gpt-5.5-xhigh': ('gpt-5.5', 'Slai-Agent; effort=xhigh', ['success_only']),
    'lite/agav/gpt-5.5': ('gpt-5.5', 'agav 0.2.0-beta.2', []),
    'lite/aiwork-code/20260909-opus-4-8-xhigh': ('claude-opus-4-8', 'AiWork.Code; reasoning_effort=xhigh', []),
    'lite/ami-agent/20260623-v0.7.0-claude-opus-4-6': ('claude-opus-4-6', 'AMI v0.7.0', ['custom_harness']),
    'lite/sapient-slingshot-agent/v2.6.0/20260629-claude-4.5': ('Claude-4.5 Sonnet', 'Sapient Slingshot 2.6.0', []),
    'lite/sapient-slingshot-agent/v3.1.0/20260817-opus-4-6': ('Claude-Opus-4-6', 'Sapient Slingshot 3.1.0', []),
    'lite/sapient-slingshot-agent/v3.3.0/20260831-gpt-5.6-sol': ('GPT-5.6-Sol', 'Sapient Slingshot 3.3.0', []),
    'lite/sapient-slingshot-agent/v3.4.0/gpt-5.6-sol': ('GPT-5.6-Sol', 'Sapient Slingshot 3.4.0', []),
    'lite/sweagent/20250501-sweagent-claude37': ('Claude Sonnet 3.7', 'SWE-agent', []),
    'lite/sweagent/20250501-sweagent-gpt41': ('GPT 4.1', 'SWE-agent', []),
    'lite/tianxicode/deepseek-flash': ('deepseek-flash', 'TianxiCode CLI v0.1.486; variant=max, thinking', []),
    # ---- multilang ----
    'multilang/cs/20260223-brokk-gpt52-flash3': ('GPT5.2 + Flash3', 'Brokk 0.22.0; planner=GPT5.2 (Medium), code agent=Flash3 (Minimal)', []),
    'multilang/cs/20260225-brokk-sonnet45-flash3': ('Sonnet4.5 + Flash3', 'Brokk 0.22.0; planner=Sonnet4.5 (Standard), code agent=Flash3 (Minimal)', []),
    'multilang/go/ami-agent/20260716-v0.7.0-claude-opus-4-6': ('Claude Opus 4.6', 'AMI v0.7.0', ['custom_harness', 'retry_on_failure']),
    'multilang/go/ami-agent/20260723-v0.7.0-claude-sonnet-4-6': ('Claude Sonnet 4.6', 'AMI v0.7.0', ['custom_harness', 'retry_on_failure']),
    'multilang/go/ami-agent/20260729-v0.7.0-claude-haiku-4-5': ('Claude Haiku 4.5', 'AMI v0.7.0', ['custom_harness', 'retry_on_failure']),
    'multilang/go/ami-agent/20260729-v0.7.0-gemini-3.1-pro-preview': ('Gemini 3.1 Pro Preview', 'AMI v0.7.0; thinking budget 65535', ['custom_harness', 'retry_on_failure']),
    'multilang/go/ami-agent/20260729-v0.7.0-gemini-3.6-flash': ('Gemini 3.6 Flash', 'AMI v0.7.0; thinking budget 65535', ['custom_harness', 'retry_on_failure']),
    'multilang/go/ami-agent/20260923-v0.7.2-qwen3.8-27b': ('Qwen/Qwen3.8-27B', 'AMI v0.7.2', []),
    'multilang/go/tianxi/deepseek-flash': ('deepseek-flash', 'TianxiCode CLI v0.1.486; variant=max, thinking', []),
    'multilang/java/ami-agent/20260817-v0.7.2-claude-opus-4-6': ('Claude Opus 4.6', 'AMI v0.7.2', ['custom_harness']),
    'multilang/java/brokk-agent/20260218-brokk-gpt52-flash3': ('GPT5.2 + Flash3', 'Brokk 0.22.0; planner=GPT5.2 (Medium), code agent=Flash3 (Minimal)', []),
    'multilang/java/brokk-agent/20260218-brokk-sonnet4.5-flash3': ('Sonnet4.5 + Flash3', 'Brokk 0.22.0; planner=Sonnet4.5 (Standard), code agent=Flash3 (Minimal)', []),
    'multilang/java/slingshot-agent/20260707-sapient-slingshot-gpt-5.5': ('GPT 5.5', 'Sapient Slingshot 2.6.0', []),
    'multilang/java/slingshot-agent/20260812-sapient-slingshot-gpt-5.5': ('GPT 5.5', 'Sapient Slingshot 3.1.0', []),
    'multilang/java/slingshot-agent/20260901-sapient-slingshot-gpt-5.6-sol': ('GPT 5.6-sol', 'Sapient Slingshot 3.3.0', []),
    'multilang/java/slingshot-agent/gpt-5.6-sol': ('GPT 5.6-sol', 'Sapient Slingshot 3.4.0; reran failed instances once', ['retry_on_failure']),
    'multilang/java/tianxicode/deepseek-flash': ('deepseek-flash', 'TianxiCode CLI v0.1.486; variant=max, thinking', []),
    'multilang/js_ts/ami-agent/20260710-v0.7.0-claude-opus-4-6/js': ('Claude Opus 4.6', 'AMI v0.7.0', ['custom_harness']),
    'multilang/js_ts/ami-agent/20260710-v0.7.0-claude-opus-4-6/ts': ('Claude Opus 4.6', 'AMI v0.7.0', ['custom_harness']),
    'multilang/js_ts/slingshot-agent/20260623-slingshot-claude4.5/js': ('Claude 4.5 Sonnet', 'Sapient Slingshot 2.6.0', []),
    'multilang/js_ts/slingshot-agent/20260623-slingshot-claude4.5/ts': ('Claude 4.5 Sonnet', 'Sapient Slingshot 2.6.0', []),
    'multilang/js_ts/slingshot-agent/20260629-sapient-slingshot-gpt-5.5/js': ('GPT-5.5', 'Sapient Slingshot 2.6.0', []),
    'multilang/js_ts/slingshot-agent/20260629-sapient-slingshot-gpt-5.5/ts': ('GPT-5.5', 'Sapient Slingshot 2.6.0', []),
    'multilang/js_ts/slingshot-agent/20260730-sapient-slingshot-gpt-5.5/js': ('GPT-5.5', 'Sapient Slingshot 3.0.0', []),
    'multilang/js_ts/slingshot-agent/20260730-sapient-slingshot-gpt-5.5/ts': ('GPT-5.5', 'Sapient Slingshot 3.0.0', []),
    'multilang/js_ts/slingshot-agent/v3.4.0/gpt-5.6-sol/js': ('GPT-5.6-Sol', 'Sapient Slingshot 3.4.0', []),
    'multilang/js_ts/slingshot-agent/v3.4.0/gpt-5.6-sol/ts': ('GPT-5.6-Sol', 'Sapient Slingshot 3.4.0', []),
    'multilang/rust/ami-agent/20260625-v0.7.0-claude-opus-4-6': ('claude-opus-4-6', 'AMI v0.7.0', ['custom_harness']),
    # all_languages: runs across all 8 languages with the scaffolds listed in the repo README
    # (SWE-agent / OpenHands / ClaudeCode forks). No README in these folders; grades are in result.json (singular).
    'multilang/all_languages/claudecode/claude-4-5-sonnet': ('claude-4-5-sonnet', 'ClaudeCode', []),
    'multilang/all_languages/claudecode/deepseek-v3.1-terminus': ('deepseek-v3.1-terminus', 'ClaudeCode', []),
    'multilang/all_languages/claudecode/deepseek-v4-pro': ('deepseek-v4-pro', 'ClaudeCode', []),
    'multilang/all_languages/claudecode/gpt-5-2-medium': ('gpt-5-2', 'ClaudeCode; effort=medium', []),
    'multilang/all_languages/claudecode/gpt-5.5-medium': ('gpt-5.5', 'ClaudeCode; effort=medium', []),
    'multilang/all_languages/openhands/deepseek-v4-pro': ('deepseek-v4-pro', 'OpenHands', []),
    'multilang/all_languages/openhands/gpt-5.5-medium': ('gpt-5.5', 'OpenHands; effort=medium', []),
    'multilang/all_languages/sweagent/claude-4-5-sonnet': ('claude-4-5-sonnet', 'SWE-agent', []),
    'multilang/all_languages/sweagent/deepseek-v3.1-terminus': ('deepseek-v3.1-terminus', 'SWE-agent', []),
    'multilang/all_languages/sweagent/deepseek-v4-pro': ('deepseek-v4-pro', 'SWE-agent', []),
    'multilang/all_languages/sweagent/gemini-3-flash': ('gemini-3-flash', 'SWE-agent', []),
    'multilang/all_languages/sweagent/gpt-5.2-medium': ('gpt-5.2', 'SWE-agent; effort=medium', []),
    'multilang/all_languages/sweagent/gpt-5.5-medium': ('gpt-5.5', 'SWE-agent; effort=medium', []),
    # ---- windows ----
    'windows/Slai-Agent/gpt-5.5-xhigh': ('gpt-5.5', 'Slai-Agent Runtime; effort=xhigh', ['success_only']),
    'windows/win-agent/deepseek-v4-pro': ('deepseek-v4-pro', 'Win-Agent', []),
    'windows/win-agent/gpt-5.5-medium': ('gpt-5.5', 'Win-Agent; effort=medium', []),
}
# model string -> (underlying model, release date, date source). mdb = project measurement-db subjects tables
# (data/mdb_full/*/formatted_tables/subjects.parquet); hf = HuggingFace model repo creation date.
MODEL_INFO = {
    'Claude Sonnet 3.7': ('Claude 3.7 Sonnet', '2025-02-24', 'mdb'),
    'GPT 4.1': ('GPT-4.1', '2025-04-14', 'mdb'),
    'Qwen3-Coder-480B-A35B': ('Qwen3-Coder-480B-A35B-Instruct', '2025-07-22', 'hf'),
    'Seed-OSS-36B-Instruct': ('Seed-OSS-36B-Instruct', '2025-08-20', 'hf'),
    'deepseek-v3.1-terminus': ('DeepSeek-V3.1-Terminus', '2025-09-22', 'general knowledge; mdb lists DeepSeek V3.1 as 2025-08-21'),
    'Claude-4.5 Sonnet': ('Claude Sonnet 4.5', '2025-09-29', 'mdb'),
    'Claude 4.5 Sonnet': ('Claude Sonnet 4.5', '2025-09-29', 'mdb'),
    'claude-4-5-sonnet': ('Claude Sonnet 4.5', '2025-09-29', 'mdb'),
    'Claude Haiku 4.5': ('Claude Haiku 4.5', '2025-10-15', 'mdb'),
    'gpt-5-2': ('GPT-5.2', '2025-12-11', 'mdb'),
    'gpt-5.2': ('GPT-5.2', '2025-12-11', 'mdb'),
    'gemini-3-flash': ('Gemini 3 Flash', '2025-12-17', 'mdb'),
    'GPT5.2 + Flash3': ('GPT-5.2 (planner) + Gemini 3 Flash (code agent)', '2025-12-17', 'mdb; later of the two'),
    'Sonnet4.5 + Flash3': ('Claude Sonnet 4.5 (planner) + Gemini 3 Flash (code agent)', '2025-12-17', 'mdb; later of the two'),
    'claude-opus-4-6': ('Claude Opus 4.6', '2026-02-05', 'mdb'),
    'Claude-Opus-4-6': ('Claude Opus 4.6', '2026-02-05', 'mdb'),
    'Claude Opus 4.6': ('Claude Opus 4.6', '2026-02-05', 'mdb'),
    'Claude Sonnet 4.6': ('Claude Sonnet 4.6', '2026-02-17', 'mdb'),
    'Gemini 3.1 Pro Preview': ('Gemini 3.1 Pro', '2026-02-19', 'mdb'),
    'gpt-5.5': ('GPT-5.5', '2026-04-23', 'mdb'),
    'GPT-5.5': ('GPT-5.5', '2026-04-23', 'mdb'),
    'GPT 5.5': ('GPT-5.5', '2026-04-23', 'mdb'),
    'deepseek-flash': ('probably DeepSeek V4 Flash', '2026-04-24', 'mdb, for DeepSeek V4 Flash; mapping is a guess'),
    'deepseek-v4-pro': ('DeepSeek V4 Pro', '2026-04-24', 'mdb'),
    'claude-opus-4-8': ('Claude Opus 4.8', '2026-05-28', 'mdb'),
    'GPT-5.6-Sol': ('GPT-5.6 Sol', '2026-06-26', 'mdb'),
    'GPT 5.6-sol': ('GPT-5.6 Sol', '2026-06-26', 'mdb'),
    'Gemini 3.6 Flash': ('Gemini 3.6 Flash', '2026 (exact date not verified; before 2026-07-29 submission)', 'not in mdb'),
    'Qwen/Qwen3.8-27B': ('Qwen3.8-27B', '2026-08-05', 'hf'),
}

# grade files: results.json, or result.json (singular) in multilang/all_languages
GRADE_FILES = {}
for name in ('results.json', 'result.json'):
    for f in glob.glob(f'{SUB_ROOT}/submissions/**/{name}', recursive=True):
        GRADE_FILES[os.path.relpath(os.path.dirname(f), f'{SUB_ROOT}/submissions')] = f
assert sorted(GRADE_FILES) == sorted(SUBS), set(GRADE_FILES) ^ set(SUBS)

res, log = [], []
for path, (model, config, flags) in SUBS.items():
    fam = path.split('/')[0]
    d = json.load(open(GRADE_FILES[path]))
    pos = set(d.get('resolved_ids', [])) | set(d.get('success_ids', []))
    neg = set(d.get('unresolved_ids', [])) | set(d.get('failure_ids', []))
    if 'resolved_ids' in d and 'success_ids' in d:  # both schemas present: must agree
        assert set(d['resolved_ids']) == set(d['success_ids']) and set(d['unresolved_ids']) == set(d['failure_ids'])
    err, inc, emp = (set(d.get(k, [])) for k in ('error_ids', 'incomplete_ids', 'empty_patch_ids'))
    assert not (pos & neg)
    # conflicting grades: graded but also listed as error/incomplete, or resolved with an empty patch
    conflict = ((pos | neg) & (err | inc)) | (pos & emp)
    # graded ids with no item text in the pinned dataset revision of this family
    no_item = (pos | neg) - fam_ids[fam]
    keep_pos, keep_neg = pos - conflict - no_item, neg - conflict - no_item
    for iid in sorted(keep_pos):
        res.append((iid, model, config, 1, 1.0, path, fam))
    for iid in sorted(keep_neg):
        res.append((iid, model, config, 1, 0.0, path, fam))
    um, rd, src = MODEL_INFO[model]
    log.append(dict(submission=path, split_family=fam, model=model, config=config, flags=';'.join(flags),
                    underlying_model=um, release_date=rd, release_2025_2026=rd.startswith(('2025', '2026')),
                    rows=len(keep_pos) + len(keep_neg), resolved=len(keep_pos), unresolved=len(keep_neg),
                    error_only=len(err - pos - neg), empty_patch_only=len(emp - pos - neg),
                    incomplete_only=len(inc - pos - neg), dropped_conflict=len(conflict), dropped_no_item=len(no_item)))
results = pd.DataFrame(res, columns=['item_key', 'model', 'config', 'trial', 'outcome', 'submission', 'split_family'])
results['trial'] = results['trial'].astype('int64')
assert not results.duplicated(['item_key', 'model', 'config', 'trial']).any()
assert set(results.item_key) <= set(items.item_key)

os.makedirs(OUT, exist_ok=True)
items.to_parquet(f'{OUT}/items.parquet', index=False)
results.to_parquet(f'{OUT}/results.parquet', index=False)
subs = pd.DataFrame(log)
subs.to_csv(f'{OUT}/submissions.csv', index=False)
json.dump(dict(n_items=len(items), items_per_family={f: len(v) for f, v in fam_ids.items()}, split_stats=split_stats,
               n_results=len(results), submissions=log), open(LOG, 'w'), indent=1)
pd.set_option('display.width', 250)
print(subs.drop(columns=['config']).to_string())
print('items', len(items), {f: len(v) for f, v in fam_ids.items()})
print('results', len(results), results.groupby('split_family').size().to_dict())
print('submissions', results.submission.nunique(), 'model strings', results.model.nunique())
print('split stats [rows, unique]', split_stats)
