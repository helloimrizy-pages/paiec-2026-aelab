"""Download the Cybench sources, then run build.py.

Usage: python -I fetch.py <download_dir> <out_dir>

Sources
- Run logs: the authors' public Google Drive folder "data_by_challenges"
  (https://drive.google.com/drive/folders/1xkA8wdAhSSYNQERQ2B7Gpzp87qP1Wgyl), fetched by the file ids in
  drive_manifest.csv from https://drive.usercontent.google.com/download?id=<id>&export=download&confirm=t.
  Drive has no revisions. The one file marked downloaded=False is blocked by Drive's malware scan and is skipped.
- task_list.txt from github.com/andyzorigin/cybench @ ca483c6443f17a20ba7d6bac3d274859a1721e51
- data/leaderboard.csv from github.com/cybench/cybench.github.io @ 3b62a0d0e3105cef73a1d835d603888328f5c2bb
Files already present are skipped, so a rerun resumes.
"""
import csv, json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DL, OUT = (os.path.abspath(p) for p in sys.argv[1:3])
MANIFEST = f"{HERE}/drive_manifest.csv"
DRIVE_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
GH = {"task_list.txt": "https://raw.githubusercontent.com/andyzorigin/cybench/ca483c6443f17a20ba7d6bac3d274859a1721e51/task_list.txt",
      "site/data_leaderboard.csv": "https://raw.githubusercontent.com/cybench/cybench.github.io/3b62a0d0e3105cef73a1d835d603888328f5c2bb/data/leaderboard.csv"}


def get(url, dest, check=None, tries=6):
    if os.path.exists(dest):
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as f:
                data = f.read()
            if check:
                check(data)
            break
        except Exception as e:
            if i == tries - 1:
                raise RuntimeError(f"{url}: {e}")
            print("retry", i, url, e, flush=True); time.sleep(3 + 5 * i)
    with open(dest + ".part", "wb") as f:
        f.write(data)
    os.replace(dest + ".part", dest)
    return len(data)


for name, url in GH.items():
    get(url, f"{DL}/{name}")
rows = list(csv.DictReader(open(MANIFEST)))
todo = [r for r in rows if r["downloaded"] == "True"]
for r in rows:
    assert not os.path.isabs(r["path"]) and ".." not in r["path"].split("/"), r["path"]
    if r["downloaded"] != "True":
        print("skipped (blocked by Drive's malware scan):", r["path"])
t0 = time.time()
with ThreadPoolExecutor(8) as ex:  # check=json.loads: Drive answers with an HTML page instead of the file on errors
    sizes = list(ex.map(lambda r: get(DRIVE_URL.format(r["drive_file_id"]), f"{DL}/drive/{r['path']}", json.loads), todo))
present = sum(os.path.exists(f"{DL}/drive/{r['path']}") for r in todo)
print(f"drive: {sum(1 for s in sizes if s)} downloaded, {sum(sizes) / 1e6:.1f} MB, {time.time() - t0:.0f} s; "
      f"{present} of {len(todo)} present", flush=True)
subprocess.run([sys.executable, "-I", f"{HERE}/build.py", f"{DL}/drive", MANIFEST, f"{DL}/task_list.txt", f"{DL}/site", OUT],
               check=True)
