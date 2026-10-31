"""Download the pinned MathVista sources and run build.py.

Usage: python -I fetch.py <download_dir> <out_dir>
Sources:
  https://github.com/lupantech/MathVista at commit c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb:
    only results/*/output_*.json and results/*/scores_*.json (sparse, partial clone of that one commit, about 120 MB)
    -> <download_dir>/MathVista
  HF dataset AI4Math/MathVista (https://huggingface.co/datasets/AI4Math/MathVista) at revision
    2b6ad69445fbb5695c9b165475e8decdbeb97747: only data/testmini-00000-of-00001-725687bf7a18d64b.parquet
    (about 142 MB) -> <download_dir>/hf
Files already present (or a checkout already at the commit) are not downloaded again.
"""
import os, subprocess, sys
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
URL, SHA = "https://github.com/lupantech/MathVista.git", "c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb"
PATTERNS = ["/results/*/output_*.json", "/results/*/scores_*.json"]
HF_REPO, HF_REV = "AI4Math/MathVista", "2b6ad69445fbb5695c9b165475e8decdbeb97747"
HF_FILE = "data/testmini-00000-of-00001-725687bf7a18d64b.parquet"


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
    sparse_checkout(URL, SHA, os.path.join(dl, "MathVista"), PATTERNS)
    hf = os.path.join(dl, "hf")
    if not os.path.exists(os.path.join(hf, HF_FILE)):
        hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset", revision=HF_REV, local_dir=hf)
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "build.py"), dl, out], check=True)


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))
