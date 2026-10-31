"""Usage: python -I fetch_drive.py <dl_dir> <ids_file>  (probe + list + fetch result files for each Drive id -> <dl_dir>/drive/)
<ids_file> lines: "<drive id> <label>". Exit status 1 if any id is not a readable zip or a step failed."""
import json, os, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
S, idsf = sys.argv[1], sys.argv[2]
PY = sys.executable
D = f"{S}/drive"
os.makedirs(f"{D}/listings", exist_ok=True)
RX = r"(^|/)(result\.txt|args\.json|all_result\.json|results\.json|README\.md)$"
ids = [l.strip().split(" ", 1) for l in open(idsf) if l.strip()]
def probe(i):
    url = f"https://drive.usercontent.google.com/download?id={i}&export=download&confirm=t"
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-3", "User-Agent": "paiec-ingest/1.0"}), timeout=60)
        return r.read(4) == b"PK\x03\x04", r.headers.get("Content-Range"), r.headers.get("Content-Disposition")
    except Exception as e:
        return False, repr(e)[:100], None
def job(t):
    i, rest = t
    ok, cr, cd = probe(i)
    json.dump(dict(id=i, label=rest, is_zip=ok, content_range=cr, content_disposition=cd), open(f"{D}/probe_{i}.json", "w"))
    if not ok:
        return i, f"skip: {rest} {cr}", 1
    url = f"https://drive.usercontent.google.com/download?id={i}&export=download&confirm=t"
    lst, out, msgs, rc = f"{D}/listings/{i}.json", f"{D}/members/{i}", [], 0
    if not os.path.exists(lst):
        p = subprocess.run([PY, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_remote_zip.py"), "list", url, lst], capture_output=True, text=True)
        msgs.append((p.stdout + p.stderr).strip()[-200:]); rc |= p.returncode
    if os.path.exists(lst):
        p = subprocess.run([PY, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_remote_zip.py"), "fetch", url, lst, RX, out, "6"], capture_output=True, text=True)
        msgs.append((p.stdout + p.stderr).strip()[-200:]); rc |= p.returncode
    return i, rest + " || " + " | ".join(msgs), rc or not os.path.exists(f"{out}/members.json")
failed = 0
with ThreadPoolExecutor(3) as ex:
    for i, msg, rc in ex.map(job, ids):
        print(i, msg, flush=True)
        failed += bool(rc)
print("ALLDONE" if not failed else f"{failed} ids failed")
sys.exit(1 if failed else 0)
