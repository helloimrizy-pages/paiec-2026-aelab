"""Download the pinned SWE-Bench Pro (public set) sources and run build.py.

Usage: python -I fetch.py <download_dir> <out_dir>
Sources (only the files build.py reads):
  GitHub https://github.com/scaleapi/SWE-bench_Pro-os @ 66f92766bba642462d4bbe5479e83f91f9211862
         helper_code/sweap_eval_full_v2.jsonl, traj/<run>/eval_results.json (9 runs); sparse, blob-filtered checkout
  HF dataset ScaleAI/SWE-bench_Pro @ 2d52cb3df914a3fcf80c7f66738b3a88ae37fc50  data/v1, data/default, data/hard
  HF dataset ScaleAI/SWE-bench_Pro @ 9c13b199fe2d3195214e2f0c6bf3c7d6f81e3877  external_hf.csv (text version sep2025)
  HF dataset ScaleAI/SWE-bench_Pro @ 8555f71a4f12cbf659c8f242769714168586cc85  external_hf.csv (text version aug2025)
  S3 https://scaleapi-results.s3.amazonaws.com/swe-bench-pro/gemini-2-5-pro-preview-250-turns-debug-nov17/output/eval_results.json
     (anonymous HTTPS; sha256 checked)
"""
import hashlib, os, shutil, subprocess, sys, urllib.request
from huggingface_hub import hf_hub_download, snapshot_download

DL, OUT = (os.path.abspath(p) for p in sys.argv[1:3])
GH_URL, GH_SHA = 'https://github.com/scaleapi/SWE-bench_Pro-os', '66f92766bba642462d4bbe5479e83f91f9211862'
GH_RUNS = ['claude-45sonnet-10132025', 'claude-4sonnet-10132025', 'gpt-5-250-turns-10132025', 'kimi-k2-instruct-10132025',
           'gptoss-paper', 'claude-opus-4-1-paper', 'claude-sonnet-4-paper', 'gemini-2-5-pro-preview-paper', 'gpt-4o-paper']
HF_REPO = 'ScaleAI/SWE-bench_Pro'
V1_REV, SEP_REV, AUG_REV = '2d52cb3df914a3fcf80c7f66738b3a88ae37fc50', '9c13b199fe2d3195214e2f0c6bf3c7d6f81e3877', '8555f71a4f12cbf659c8f242769714168586cc85'
S3_URL = 'https://scaleapi-results.s3.amazonaws.com/swe-bench-pro/gemini-2-5-pro-preview-250-turns-debug-nov17/output/eval_results.json'
S3_SHA256 = '53955fcef2ae0dded3c017ac7258b0625c55d183925e513cfd7a2cdf29de6866'
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


def sha256(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


os.makedirs(DL, exist_ok=True); os.makedirs(OUT, exist_ok=True)
repo = f'{DL}/SWE-bench_Pro-os'
sparse_checkout(repo, GH_URL, GH_SHA, ['/helper_code/sweap_eval_full_v2.jsonl'] + [f'/traj/{r}/eval_results.json' for r in GH_RUNS])
hf = snapshot_download(HF_REPO, repo_type='dataset', revision=V1_REV, local_dir=f'{DL}/hf_{V1_REV[:8]}',
                       allow_patterns=['data/v1/*.parquet', 'data/default/*.parquet', 'data/hard/*.parquet'])
sep_csv = hf_hub_download(HF_REPO, 'external_hf.csv', repo_type='dataset', revision=SEP_REV, local_dir=f'{DL}/hf_{SEP_REV[:8]}')
aug_csv = hf_hub_download(HF_REPO, 'external_hf.csv', repo_type='dataset', revision=AUG_REV, local_dir=f'{DL}/hf_{AUG_REV[:8]}')
s3 = f'{DL}/s3/gemini-2-5-pro-preview-250-turns-debug-nov17/eval_results.json'
if not (os.path.exists(s3) and sha256(s3) == S3_SHA256):
    os.makedirs(os.path.dirname(s3), exist_ok=True)
    print('downloading', S3_URL)
    with urllib.request.urlopen(S3_URL, timeout=120) as r, open(s3 + '.part', 'wb') as f:
        shutil.copyfileobj(r, f)
    os.replace(s3 + '.part', s3)
assert sha256(s3) == S3_SHA256, 'S3 eval_results.json changed upstream'
build = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build.py')
subprocess.run([sys.executable, '-I', build, repo, hf, sep_csv, aug_csv, s3, OUT], check=True)
