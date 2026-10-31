"""Usage: python -I fetch_hf_list.py <dl_dir> [zip ...]  (central directory of each HF zip -> <dl_dir>/listings/<zip>.json)
Zips default to every .zip in <dl_dir>/tree_verified.json. Exit status 1 if any listing failed."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
R = "5473c39e42a538a187a9b2c2b499db59d560fd8c"
S = sys.argv[1]
PY = sys.executable
t = json.load(open(f"{S}/tree_verified.json"))
files = sys.argv[2:] or [x["path"] for x in t if x["type"] == "file" and x["path"].endswith(".zip")]
os.makedirs(f"{S}/listings", exist_ok=True)
def job(f):
    out = f"{S}/listings/{f}.json"
    if os.path.exists(out):
        return f, "cached", 0
    url = f"https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs/resolve/{R}/{f}"
    p = subprocess.run([PY, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_remote_zip.py"), "list", url, out], capture_output=True, text=True)
    return f, (p.stdout + p.stderr).strip()[-300:], p.returncode
failed = 0
with ThreadPoolExecutor(6) as ex:
    for f, msg, rc in ex.map(job, files):
        print(f, "|", msg, flush=True)
        failed += rc != 0
sys.exit(1 if failed else 0)
