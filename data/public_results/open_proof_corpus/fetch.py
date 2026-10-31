"""Download the pinned Open Proof Corpus (OPC) snapshot and run build.py.

Usage: python -I fetch.py <download_dir> <out_dir>
Source: HF dataset INSAIT-Institute/OPC (https://huggingface.co/datasets/INSAIT-Institute/OPC)
at revision dcc3b4804e2d126ea34b13e3e0cd998c3302644b. Only the six split parquet files under data/ are
downloaded (about 23 MB). Files already present are not downloaded again.
"""
import glob, os, subprocess, sys
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
REPO, REV = "INSAIT-Institute/OPC", "dcc3b4804e2d126ea34b13e3e0cd998c3302644b"
SPLITS = ["generic", "test", "putnambench", "matharena", "pass_at_n", "best_of_n"]


def main(dl, out):
    snap = os.path.join(dl, "OPC")
    pats = [f"data/{s}-*.parquet" for s in SPLITS]
    if not all(glob.glob(os.path.join(snap, p)) for p in pats):
        snapshot_download(REPO, repo_type="dataset", revision=REV, allow_patterns=pats, local_dir=snap)
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "build.py"), snap, out], check=True)


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))
