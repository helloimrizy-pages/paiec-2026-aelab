"""Download the pinned SWE-bench-Live sources and run build.py.

Usage: python -I fetch.py <download_dir> <out_dir>
Sources (only the files build.py reads):
  GitHub https://github.com/SWE-bench-Live/submission @ 641b727c8a5e3e5d0af08a348e632b12c3f11aef
         submissions/**/results.json and submissions/**/result.json; sparse, blob-filtered checkout
  HF dataset SWE-bench-Live/SWE-bench-Live @ b51a86422e10cfd403beb4773e5a2947953e36ec  data/{lite,verified,test,full}-*.parquet
  HF dataset SWE-bench-Live/MultiLang @ 3638632e8153a10ca422c1022bed79023084b5c9  data/{c,cpp,cs,go,java,js,rust,ts}-*.parquet
  HF dataset SWE-bench-Live/Windows @ b9fa25f2732dbc2c93b9738b610d053d8b7bd7a9  data/test-*.parquet
build.py writes items.parquet, results.parquet, submissions.csv to <out_dir> and its summary log to <download_dir>/build_log.json.
"""
import os, subprocess, sys
from huggingface_hub import snapshot_download

DL, OUT = (os.path.abspath(p) for p in sys.argv[1:3])
GH_URL, GH_SHA = 'https://github.com/SWE-bench-Live/submission', '641b727c8a5e3e5d0af08a348e632b12c3f11aef'
HF = [  # (dataset, revision, splits)
    ('SWE-bench-Live/SWE-bench-Live', 'b51a86422e10cfd403beb4773e5a2947953e36ec', ['lite', 'verified', 'test', 'full']),
    ('SWE-bench-Live/MultiLang', '3638632e8153a10ca422c1022bed79023084b5c9', ['c', 'cpp', 'cs', 'go', 'java', 'js', 'rust', 'ts']),
    ('SWE-bench-Live/Windows', 'b9fa25f2732dbc2c93b9738b610d053d8b7bd7a9', ['test']),
]
ENV = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')


def git(*a, cwd):
    return subprocess.run(['git', '-c', 'advice.detachedHead=false', *a], cwd=cwd, env=ENV, check=True, capture_output=True, text=True).stdout.strip()


def sparse_checkout(d, url, sha, patterns):
    if os.path.isdir(f'{d}/.git'):
        try:
            if git('rev-parse', 'HEAD', cwd=d) == sha:
                print(d, 'present at', sha); return
        except subprocess.CalledProcessError:
            pass
    else:
        os.makedirs(d, exist_ok=True)
        git('init', '-q', cwd=d)
        git('remote', 'add', 'origin', url, cwd=d)
    git('sparse-checkout', 'set', '--no-cone', *patterns, cwd=d)
    print('fetching', url, sha)
    git('fetch', '-q', '--depth', '1', '--filter=blob:none', 'origin', sha, cwd=d)
    git('checkout', '-q', sha, cwd=d)
    assert git('rev-parse', 'HEAD', cwd=d) == sha


os.makedirs(DL, exist_ok=True); os.makedirs(OUT, exist_ok=True)
sub = f'{DL}/submission'
sparse_checkout(sub, GH_URL, GH_SHA, ['/submissions/**/results.json', '/submissions/**/result.json'])
dirs = [snapshot_download(ds, repo_type='dataset', revision=rev, local_dir=f'{DL}/{ds.split("/")[1]}',
                          allow_patterns=[f'data/{s}-*.parquet' for s in splits]) for ds, rev, splits in HF]
build = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build.py')
subprocess.run([sys.executable, '-I', build, sub, *dirs, OUT, f'{DL}/build_log.json'], check=True)
