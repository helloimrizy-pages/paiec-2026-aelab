"""Build normalized Code2Bench-2509 per-item results (SC-Python and WSC-Python).

Outputs items.parquet, results.parquet and runs.csv.

Usage: python -I build.py <repo_clone> <out_dir>
  repo_clone  checkout of https://github.com/Code2Bench/Code2Bench at 92fbd3da2eec702184170268fe7cb92fbaab28ec
              (code2bench-2509/Python and code2bench-2509/weakly; a sparse checkout from fetch.py is enough)

Outcome = 1.0 when the task's status in transformed_status.json is SUCCESS (the generated function
passed every property-based test case), else 0.0. One run per model and task (trial 1).
"""
import json, os, sys, glob, re
import pandas as pd

REPO, OUT = sys.argv[1:3]
SHA = '92fbd3da2eec702184170268fe7cb92fbaab28ec'
ROOT = f'{REPO}/code2bench-2509'
# folder -> (family label used in item_key and paper/site, index file)
FAMS = {'Python': ('SC-Python', 'index.json'), 'weakly': ('WSC-Python', 'weakly_index.json')}

# ---------------- items ----------------
items = []
for folder, (fam, idx) in FAMS.items():
    index = json.load(open(f'{ROOT}/{folder}/{idx}'))
    dirs = sorted([d for d in os.listdir(f'{ROOT}/{folder}') if d.isdigit()], key=int)
    for d in dirs:
        with open(f'{ROOT}/{folder}/{d}/instruction.txt', encoding='utf-8') as f:
            text = f.read()   # the user message, verbatim (CodeGenerationRunner.prepare_llm_input)
        fn = re.findall(r'def (\w+)\(', text)
        src = index.get(d)
        if src is not None:    # index entry names the source function; check it agrees with the instruction
            assert fn and fn[-1] in src.split('.'), (folder, d)
        extra = {'family': fam, 'repo_folder': f'code2bench-2509/{folder}/{d}', 'function_name': fn[-1] if fn else None,
                 'source_function': src}
        items.append({'item_key': f'{fam}/{d}', 'text': text, 'extra': extra})
items = pd.DataFrame(items)
assert items.item_key.is_unique
# mark items whose instruction text is shared with another item
dup = items.groupby('text').item_key.apply(list)
dup = {k: v for v in dup if len(v) > 1 for k in v}
items['extra'] = [json.dumps({**e, **({'same_text_as': [x for x in dup[k] if x != k]} if k in dup else {})}, ensure_ascii=False)
                  for k, e in zip(items.item_key, items.extra)]
keys = set(items.item_key)

# ---------------- results ----------------
rows, runs = [], []
for folder, (fam, _) in FAMS.items():
    stats = json.load(open(f'{ROOT}/{folder}/statistics.json'))['Statistics'][0]
    for f in sorted(glob.glob(f'{ROOT}/{folder}/**/transformed_status.json', recursive=True)):
        d = json.load(open(f))
        assert len(d) == 1
        (model, lst), = d.items()
        # weakly files hold a list of 10 entries; all 10 are identical copies (checked here), so they are one run
        assert all(e == lst[0] for e in lst), f
        entry = lst[0]
        n_succ = sum(v['status'] == 'SUCCESS' for v in entry.values())
        # which statistics.json run (same model, same total) has the same Success count
        match = [r for r in stats.get(model, []) if r['Total'] == len(entry) and r['Success'] == n_succ]
        assert match, (f, n_succ)
        final = stats[model][-1]
        is_final = final['Total'] == len(entry) and final['Success'] == n_succ
        for iid, v in sorted(entry.items(), key=lambda kv: int(kv[0])):
            key = f'{fam}/{iid}'
            assert key in keys, key
            assert v['status'] == 'SUCCESS' or v.get('pass_rate', 0) < 1.0
            rows.append({'item_key': key, 'model': model,
                         'config': f'Code2Bench-2509 {fam}; zero-shot, greedy decoding (temperature 0), Pass@1',
                         'trial': 1, 'outcome': 1.0 if v['status'] == 'SUCCESS' else 0.0,
                         'status': v['status'], 'pass_rate': v.get('pass_rate'),
                         'submission': os.path.relpath(f, REPO),
                         # not_final_run: per-item file matches an earlier statistics.json run, not the run behind the paper's Table 3
                         'flags': '' if is_final else 'not_final_run'})
        runs.append({'submission': os.path.relpath(f, REPO), 'family': fam, 'model': model, 'n_items': len(entry),
                     'n_success': n_succ, 'copies_in_file': len(lst),
                     'matching_statistics_runs': '; '.join(f"{r['Time']} ({r['Success']}/{r['Total']})" for r in match),
                     'final_statistics_run': f"{final['Time']} ({final['Success']}/{final['Total']})",
                     'matches_final_run': is_final})
res = pd.DataFrame(rows)
res['trial'] = res['trial'].astype(int)
res['pass_rate'] = res['pass_rate'].astype(float)
assert not res.duplicated(['item_key', 'model']).any()
os.makedirs(OUT, exist_ok=True)
items.to_parquet(f'{OUT}/items.parquet', index=False)
res.to_parquet(f'{OUT}/results.parquet', index=False)
pd.DataFrame(runs).to_csv(f'{OUT}/runs.csv', index=False)
print(len(items), 'items;', len(res), 'results;', res.model.nunique(), 'models;', res.item_key.nunique(), 'items with results')
print('items sharing text:', dup)
print(pd.DataFrame(runs).drop(columns=['submission']).to_string())
