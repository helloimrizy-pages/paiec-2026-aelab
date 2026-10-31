"""Download the pinned LiveCodeBench Pro problem statements, then run build.py.

Usage: python -I fetch.py <download_dir> <out_dir>

Sources:
  https://huggingface.co/datasets/QAQAQAQAQ/LiveCodeBench-Pro @ adebffce047dddb7768a86bace6aea4f7425e3bc
      data/*.parquet (six splits, about 1.6 MB). Gated with automatic approval: the Hugging Face account
      whose token huggingface_hub finds must have requested access once.
  api_snapshot.json.gz (shipped next to this script): the leaderboard backend https://webhook.cp-bench.orzzh.com
      as served on 2026-10-19T18:47Z. The backend is not versioned, so the shipped snapshot is used.
      fetch_api.py makes a new snapshot; pass it to build.py directly to build from it.
Files already downloaded are skipped, so a rerun resumes.
"""
import os, subprocess, sys
from huggingface_hub import snapshot_download

REPO, REV = "QAQAQAQAQ/LiveCodeBench-Pro", "adebffce047dddb7768a86bace6aea4f7425e3bc"
HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    dl, out = (os.path.abspath(a) for a in sys.argv[1:3])
    os.makedirs(out, exist_ok=True)
    snapshot_download(repo_id=REPO, repo_type="dataset", revision=REV, allow_patterns=["data/*.parquet"],
                      local_dir=f"{dl}/hf")
    subprocess.run([sys.executable, "-I", f"{HERE}/build.py", f"{HERE}/api_snapshot.json.gz", f"{dl}/hf/data", out],
                   check=True, cwd=out)
