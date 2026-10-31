"""Usage: python -I fetch_hf_members.py <dl_dir> [zip ...]  (small result files of each HF zip -> <dl_dir>/members/<zip>/members.json)
Needs <dl_dir>/listings/<zip>.json. Zips default to every .zip in <dl_dir>/tree_verified.json. Exit status 1 if any zip failed."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
R = "5473c39e42a538a187a9b2c2b499db59d560fd8c"
S = sys.argv[1]
PY = sys.executable
RX = r"(^|/)(result\.txt|args\.json|all_result\.json|results\.json|manifest\.json|README\.md|status\.json|final_summary\.json|merged_summary_with_repair\.json|phase_results\.json|test_[^/]*\.json|SHA256SUMS|result\.json|LICENSE)$"
t = json.load(open(f"{S}/tree_verified.json"))
files = sys.argv[2:] or [x["path"] for x in t if x["type"] == "file" and x["path"].endswith(".zip")]
def job(f):
    out = f"{S}/members/{f}"
    url = f"https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs/resolve/{R}/{f}"
    p = subprocess.run([PY, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_remote_zip.py"), "fetch", url, f"{S}/listings/{f}.json", RX, out, "12"], capture_output=True, text=True)
    return f, (p.stdout + p.stderr).strip()[-400:], p.returncode
failed = 0
with ThreadPoolExecutor(4) as ex:
    for f, msg, rc in ex.map(job, files):
        print(f, "|", msg, flush=True)
        failed += rc != 0
print("ALLDONE" if not failed else f"{failed} zips failed")
sys.exit(1 if failed else 0)
