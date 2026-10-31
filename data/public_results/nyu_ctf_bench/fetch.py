"""Download the pinned NYU CTF Bench sources and run build.py.

Usage: python -I fetch.py <download_dir> <out_dir>
Sparse, blob-filtered git checkouts at pinned commits (only the files build.py reads):
  https://github.com/NYU-LLM-CTF/leaderboard_submissions @ 69def02a79e8b1d890e1e0d5c11106a9f40e11e1  transcripts/*/summary.json
  https://github.com/NYU-LLM-CTF/NYU_CTF_Bench @ 4c5744f55a31a58e03d6118c9dbb1c18e5bbdfa9  test_dataset.json, test/**/challenge.json
"""
import os, subprocess, sys

DL, OUT = (os.path.abspath(p) for p in sys.argv[1:3])
REPOS = [
    ('leaderboard_submissions', 'https://github.com/NYU-LLM-CTF/leaderboard_submissions', '69def02a79e8b1d890e1e0d5c11106a9f40e11e1',
     ['/transcripts/*/summary.json']),
    ('NYU_CTF_Bench', 'https://github.com/NYU-LLM-CTF/NYU_CTF_Bench', '4c5744f55a31a58e03d6118c9dbb1c18e5bbdfa9',
     ['/test_dataset.json', '/test/**/challenge.json']),
]
ENV = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')


def git(*a, cwd):
    return subprocess.run(['git', '-c', 'advice.detachedHead=false', *a], cwd=cwd, env=ENV, check=True, capture_output=True, text=True).stdout.strip()


def sparse_checkout(name, url, sha, patterns):
    d = f'{DL}/{name}'
    if os.path.isdir(f'{d}/.git'):
        try:
            if git('rev-parse', 'HEAD', cwd=d) == sha:
                print(name, 'present at', sha); return
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
for r in REPOS:
    sparse_checkout(*r)
build = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build.py')
subprocess.run([sys.executable, '-I', build, DL, OUT], check=True)
