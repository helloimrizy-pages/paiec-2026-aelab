"""Download the pinned Code2Bench-2509 release files, then run build.py.

Usage: python -I fetch.py <download_dir> <out_dir>

Source: https://github.com/Code2Bench/Code2Bench @ 92fbd3da2eec702184170268fe7cb92fbaab28ec
  sparse checkout of code2bench-2509/{Python,weakly}/: index files, statistics.json,
  <task>/instruction.txt and every transformed_status.json (about 1.5 MB).
The repo is skipped if the pinned commit is already checked out, so a rerun resumes.
"""
import os, subprocess, sys

URL = "https://github.com/Code2Bench/Code2Bench"
SHA = "92fbd3da2eec702184170268fe7cb92fbaab28ec"
PATTERNS = [f"/code2bench-2509/{folder}/{p}" for folder, idx in [("Python", "index.json"), ("weakly", "weakly_index.json")]
            for p in [idx, "statistics.json", "*/instruction.txt", "**/transformed_status.json"]]
HERE = os.path.dirname(os.path.abspath(__file__))


def git(d, *a):
    subprocess.run(["git", "-C", d, *a], check=True)


def fetch_repo(d):
    if not os.path.isdir(f"{d}/.git"):
        os.makedirs(d, exist_ok=True)
        git(d, "init", "-q")
        git(d, "remote", "add", "origin", URL)
        git(d, "config", "core.sparseCheckout", "true")
    with open(f"{d}/.git/info/sparse-checkout", "w") as f:
        f.write("\n".join(PATTERNS) + "\n")
    if subprocess.run(["git", "-C", d, "cat-file", "-e", f"{SHA}^{{commit}}"], capture_output=True).returncode:
        git(d, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", SHA)
    git(d, "checkout", "-q", "--detach", SHA)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    dl, out = (os.path.abspath(a) for a in sys.argv[1:3])
    os.makedirs(out, exist_ok=True)
    fetch_repo(f"{dl}/Code2Bench")
    subprocess.run([sys.executable, "-I", f"{HERE}/build.py", f"{dl}/Code2Bench", out], check=True, cwd=out)
