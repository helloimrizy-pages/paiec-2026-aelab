"""Download the pinned InteractScience release files and run build.py.

Usage: python -I fetch.py <download_dir> <out_dir>
Source: https://github.com/open-compass/InteractScience at commit c21ea8dffad175a43d810fa027a152113956e6ff.
Only the files build.py reads are checked out (sparse, partial clone of that one commit, about 300 MB):
data/interactscience.jsonl, the Playwright JSON reports under results/lm_results/PFT_test_results/ and
results/vlm_results/component_test_results/, and PFT_tests/*.spec.js.
A rerun reuses a checkout that is already at the commit.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
URL, SHA = "https://github.com/open-compass/InteractScience.git", "c21ea8dffad175a43d810fa027a152113956e6ff"
PATTERNS = ["/data/interactscience.jsonl", "/PFT_tests/*.spec.js",
            "/results/lm_results/PFT_test_results/*_component_results.json",
            "/results/vlm_results/component_test_results/*_component_results.json"]


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], check=True, text=True, capture_output=True).stdout.strip()


def sparse_checkout(url, sha, repo, patterns):
    """Fetch only commit `sha` without blobs, then check out the paths that match `patterns`."""
    os.makedirs(repo, exist_ok=True)
    git(repo, "init", "-q")
    if "origin" not in git(repo, "remote").split():
        git(repo, "remote", "add", "origin", url)
    git(repo, "sparse-checkout", "set", "--no-cone", *patterns)
    if subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode:
        git(repo, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha)
    git(repo, "reset", "-q", "--hard", sha)
    assert git(repo, "rev-parse", "HEAD") == sha


def main(dl, out):
    repo = os.path.join(dl, "InteractScience")
    sparse_checkout(URL, SHA, repo, PATTERNS)
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "build.py"), repo, out], check=True)


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))
