"""Build normalized SWE-Bench Pro (public set) per-item results.

Outputs items.parquet, results.parquet and runs.csv.

Two task-text versions exist, because the released runs were made on two different public dataset versions:
  text_version 'sep2025': the 731-task public set as on HF from 2025-09-18 to 2026-02-09 (external_hf.csv @ 9c13b199).
                          Agents saw  "{problem_statement}\n\nRequirements:\n{requirements}\n\nNew interfaces introduced:\n{interface}".
                          item_key = instance_id. Equal to the HF 'v1' config text except for 5 tasks.
  text_version 'aug2025': the 851-task first release on HF (external_hf.csv @ 8555f71a, 2025-08-27).
                          Agents saw  "{problem_statement}\n\n{requirements}".
                          item_key = instance_id + '#aug2025'. The four "paper" runs (Opus 4.1, Sonnet 4, Gemini, GPT-4o) used it.

Outcome: each run's eval_results.json maps instance_id to true (resolved: 1.0) or false (0.0); resolved means every
fail_to_pass and pass_to_pass test passes. Graded IDs with no text in the run's text version get no row.

Usage: python -I build.py <repo_clone> <hf_v1_snapshot> <sep_csv> <aug_csv> <s3_nov17_eval_results.json> <out_dir>
  repo_clone       checkout of https://github.com/scaleapi/SWE-bench_Pro-os at 66f92766bba642462d4bbe5479e83f91f9211862
  hf_v1_snapshot   snapshot of HF dataset ScaleAI/SWE-bench_Pro at 2d52cb3df914a3fcf80c7f66738b3a88ae37fc50
                   (data/v1 for the v1 cross-check; data/default and data/hard for the in_v2 / hard51 flags)
  sep_csv          ScaleAI/SWE-bench_Pro external_hf.csv at revision 9c13b199fe2d3195214e2f0c6bf3c7d6f81e3877
                   (sha256 d2823b3c45341ac43586f49c9e49ee34f9fb10c4f5eddca40289750439513c2a)
  aug_csv          ScaleAI/SWE-bench_Pro external_hf.csv at revision 8555f71a4f12cbf659c8f242769714168586cc85
                   (sha256 8a908cd82f5a30e182275ab4e7f7364d62b284c3dc45882ce08a36601c201b44)
  s3_nov17...      https://scaleapi-results.s3.amazonaws.com/swe-bench-pro/gemini-2-5-pro-preview-250-turns-debug-nov17/output/eval_results.json
                   (ETag/MD5 af1a015f5dad97c227d4fb380296afea, LastModified 2025-11-17T22:28:54Z)
"""
import hashlib, json, os, sys
import pandas as pd

REPO, HF, SEP_CSV, AUG_CSV, S3_NOV17, OUT = sys.argv[1:7]
GH_SHA = '66f92766bba642462d4bbe5479e83f91f9211862'
SEP_REV, AUG_REV, V1_REV = '9c13b199fe2d3195214e2f0c6bf3c7d6f81e3877', '8555f71a4f12cbf659c8f242769714168586cc85', '2d52cb3df914a3fcf80c7f66738b3a88ae37fc50'
for p, h in [(SEP_CSV, 'd2823b3c45341ac43586f49c9e49ee34f9fb10c4f5eddca40289750439513c2a'),
             (AUG_CSV, '8a908cd82f5a30e182275ab4e7f7364d62b284c3dc45882ce08a36601c201b44'),
             (S3_NOV17, '53955fcef2ae0dded3c017ac7258b0625c55d183925e513cfd7a2cdf29de6866')]:
    assert hashlib.sha256(open(p, 'rb').read()).hexdigest() == h, p


def tmpl_sep(r):
    # verbatim template of helper_code/create_problem_statement.py; pandas reads empty CSV cells as NaN and the
    # 2025 pipeline printed them as the string 'nan' (seen in trajectories), so str() reproduces that.
    return f"""{str(r['problem_statement'])}

Requirements:
{str(r['requirements'])}

New interfaces introduced:
{str(r['interface'])}"""


def tmpl_aug(r):
    return f"{r['problem_statement']}\n\n{r['requirements']}"


def plain(s):
    # 328 of 731 rows store text fields as a JSON-quoted string literal (leading '"', escaped '\\n'); the agent saw
    # that quoted form. Decode it for an extra plain-text copy.
    if isinstance(s, str) and s.startswith('"'):
        try:
            x = json.loads(s)
            if isinstance(x, str):
                return x
        except Exception:
            pass
    return s


v1 = pd.read_parquet(f'{HF}/data/v1/test-00000-of-00001.parquet').set_index('instance_id')
v2_ids = set(pd.read_parquet(f'{HF}/data/default/test-00000-of-00001.parquet').instance_id)
hard_ids = set(pd.read_parquet(f'{HF}/data/hard/test-00000-of-00001.parquet').instance_id)
sep = pd.read_csv(SEP_CSV)
aug = pd.read_csv(AUG_CSV)
assert len(sep) == 731 and sep.instance_id.is_unique and set(sep.instance_id) == set(v1.index)
assert len(aug) == 851 and aug.instance_id.is_unique and aug[['problem_statement', 'requirements']].notna().all().all()

helper = {}
with open(f'{REPO}/helper_code/sweap_eval_full_v2.jsonl') as f:
    for line in f:
        r = json.loads(line)
        helper[r['instance_id']] = r['problem_statement']

items, differs_v1 = [], []
for _, r in sep.iterrows():
    iid = r.instance_id
    text = tmpl_sep(r)
    assert helper[iid].rstrip() == text.rstrip(), iid          # repo's own formatted instance file agrees
    v1_text = tmpl_sep(v1.loc[iid])
    extra = {'text_version': 'sep2025', 'hf_dataset': 'ScaleAI/SWE-bench_Pro', 'hf_file': 'external_hf.csv', 'hf_revision': SEP_REV,
             'repo': r.repo, 'repo_language': r.repo_language, 'base_commit': r.base_commit,
             'issue_specificity': r.issue_specificity, 'issue_categories': r.issue_categories,
             'problem_statement_plain': plain(v1.at[iid, 'problem_statement']),
             'in_v2': iid in v2_ids, 'hard51': iid in hard_ids, 'text_equals_hf_v1_config': text == v1_text}
    if text != v1_text:
        differs_v1.append(iid)
        extra['hf_v1_config_text'] = v1_text
    items.append({'item_key': iid, 'text': text, 'extra': json.dumps(extra, ensure_ascii=False)})
assert len(differs_v1) == 5, differs_v1
for _, r in aug.iterrows():
    extra = {'text_version': 'aug2025', 'hf_dataset': 'ScaleAI/SWE-bench_Pro', 'hf_file': 'external_hf.csv', 'hf_revision': AUG_REV,
             'instance_id': r.instance_id, 'repo': r.repo, 'repo_language': r.repo_language, 'base_commit': r.base_commit,
             'in_final_731': r.instance_id in v1.index, 'in_v2': r.instance_id in v2_ids}
    items.append({'item_key': r.instance_id + '#aug2025', 'text': tmpl_aug(r), 'extra': json.dumps(extra, ensure_ascii=False)})
items = pd.DataFrame(items)
assert items.item_key.is_unique
keys = set(items.item_key)

# ---------------- runs ----------------
# run folder -> (model display name as released, config, flags, source, text_version)
# Display names: leaderboard page (index.html) for the dated runs and gpt-oss; paper Table 1 names for the paper runs.
# Config: SWE-agent settings read from the run's trajectory files on S3 (one to seven trajectories sampled per run).
LB = 'SWE-agent 1.1.0; leaderboard run 2025-10-13; 250 calls, no per-instance cost cap'
RUNS = {
    'claude-45sonnet-10132025': ('claude-4-5-Sonnet', LB, [], 'github', 'sep2025'),
    'claude-4sonnet-10132025': ('claude-4-Sonnet', LB, [], 'github', 'sep2025'),
    'gpt-5-250-turns-10132025': ('gpt-5-2025-08-07 (High)', LB, [], 'github', 'sep2025'),
    'kimi-k2-instruct-10132025': ('kimi-k2-instruct', LB, ['text_version_unverified'], 'github', 'sep2025'),
    'gptoss-paper': ('gpt-oss-120b', 'SWE-agent 1.1.0; folder says paper run; trajectory config: 50 calls, $1 per-instance cost cap', [], 'github', 'sep2025'),
    'gemini-2-5-pro-preview-250-turns-debug-nov17': ('gemini-2.5-pro-preview-06-05', 'SWE-agent 1.1.0; debug run 2025-11-17; 250 calls, no per-instance cost cap', ['debug_run', 's3_only'], 's3', 'sep2025'),
    'claude-opus-4-1-paper': ('Claude Opus 4.1', 'SWE-agent 1.1.0; paper run; trajectory config: 100 calls, $4 per-instance cost cap', ['label_disputed', 'near_duplicate'], 'github', 'aug2025'),
    'claude-sonnet-4-paper': ('Claude Sonnet 4', 'SWE-agent 1.1.0; paper run; trajectory config: 100 calls, $4 per-instance cost cap', ['label_disputed', 'near_duplicate'], 'github', 'aug2025'),
    'gemini-2-5-pro-preview-paper': ('Gemini 2.5 Pro Preview', 'SWE-agent 1.1.0; paper run; trajectory config: 100 calls, $2 per-instance cost cap', [], 'github', 'aug2025'),
    'gpt-4o-paper': ('OpenAI GPT-4o', 'SWE-agent 1.1.0; paper run; trajectory config: 100 calls, $4 per-instance cost cap', [], 'github', 'aug2025'),
}
# run folder -> (model id in the run's SWE-agent config on S3, release date)
MODEL_INFO = {
    'claude-45sonnet-10132025': ('anthropic/claude-sonnet-4-5', '2025-09-29'),
    'claude-4sonnet-10132025': ('anthropic/claude-sonnet-4-20250514', '2025-05-22'),
    'gpt-5-250-turns-10132025': ('openai/gpt-5', '2025-08-07'),
    'kimi-k2-instruct-10132025': ('not checked: no S3 folder with this name', '2025-07-11'),
    'gptoss-paper': ('fireworks_ai/gpt-oss-120b', '2025-08-05'),
    'gemini-2-5-pro-preview-250-turns-debug-nov17': ('gemini/gemini-2.5-pro-preview-06-05', '2025-06-05'),
    'claude-opus-4-1-paper': ('anthropic/claude-4-sonnet-20250514 (sic)', '2025-08-05 if Opus 4.1; 2025-05-22 if Sonnet 4'),
    'claude-sonnet-4-paper': ('anthropic/claude-opus-4-1-20250805 (sic)', '2025-05-22 if Sonnet 4; 2025-08-05 if Opus 4.1'),
    'gemini-2-5-pro-preview-paper': ('gemini/gemini-2.5-pro-preview-06-05', '2025-06-05'),
    'gpt-4o-paper': ('openai/gpt-4o', '2024-05-13'),
}

rows, runs = [], []
for run, (model, config, flags, src, tv) in RUNS.items():
    path = f'{REPO}/traj/{run}/eval_results.json' if src == 'github' else S3_NOV17
    grades = json.load(open(path))
    assert isinstance(grades, dict) and all(isinstance(v, bool) for v in grades.values()), run
    suffix = '' if tv == 'sep2025' else '#aug2025'
    kept = {k + suffix: v for k, v in grades.items() if k + suffix in keys}
    for key, ok in sorted(kept.items()):
        rows.append({'item_key': key, 'model': model, 'config': config, 'trial': 1, 'outcome': 1.0 if ok else 0.0,
                     'submission': run, 'source': src, 'text_version': tv, 'flags': ','.join(flags)})
    mid, rel = MODEL_INFO[run]
    runs.append({'submission': run,
                 'source': (f'github scaleapi/SWE-bench_Pro-os@{GH_SHA}:traj/{run}/eval_results.json' if src == 'github'
                            else 's3://scaleapi-results/swe-bench-pro/gemini-2-5-pro-preview-250-turns-debug-nov17/output/eval_results.json'),
                 'model': model, 'model_id_in_run_config': mid, 'release_date': rel, 'release_2025_2026': not rel.startswith('2024'),
                 'config': config, 'text_version': tv, 'flags': ','.join(flags),
                 'n_graded': len(grades), 'n_rows': len(kept), 'n_dropped_no_text': len(grades) - len(kept),
                 'n_resolved': int(sum(kept.values())), 'n_resolved_in_final_731': int(sum(v for k, v in grades.items() if k in v1.index))})
res = pd.DataFrame(rows)
res['trial'] = res['trial'].astype(int)
assert not res.duplicated(['item_key', 'submission']).any()
os.makedirs(OUT, exist_ok=True)
items.to_parquet(f'{OUT}/items.parquet', index=False)
res.to_parquet(f'{OUT}/results.parquet', index=False)
pd.DataFrame(runs).to_csv(f'{OUT}/runs.csv', index=False)
print(len(items), 'items;', len(res), 'results;', res.model.nunique(), 'models;', res.item_key.nunique(), 'items with >=1 result')
print('sep2025 texts differing from HF v1 config:', differs_v1)
print(pd.DataFrame(runs)[['submission', 'text_version', 'n_graded', 'n_rows', 'n_dropped_no_text', 'n_resolved', 'n_resolved_in_final_731']].to_string())
