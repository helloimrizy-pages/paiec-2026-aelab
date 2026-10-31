#!/usr/bin/env python3
"""Download the OSWorld provider files that build.py reads, then run build.py.

Usage:  python -I fetch.py <download_dir> <out_dir>

Sources (only small files are kept; files already present are skipped, so a rerun resumes):
  HF dataset xlangai/ubuntu_osworld_verified_trajs @ 5473c39e42a538a187a9b2c2b499db59d560fd8c
    zip central directories and small result files by HTTP range requests (fetch_hf_list.py, fetch_hf_members.py,
    fetch_remote_zip.py); oswv_claude-sonnet-4-6.tar.gz and the tar.gz inside coact1-150-100-50-15steps.zip are
    streamed once, keeping only small files (fetch_stream_tar.py)
  OS-World/OS-World.github.io @ 62f8466dbe8b4d67c104b5ead3f8271da01aa09d: static/data/osworld_verified_results.xlsx
  xlang-ai/OSWorld @ 9f4b4d34319589f3ba6947da6045b0955b08900e: blobless bare clone plus the task-file blobs
    (evaluation_examples/examples/*/*.json, test_all.json, test_nogdrive.json) of every first-parent commit
  Google Drive folder 1XtGJe3NYbceLh58qHqXId81_d5rwFLwQ (not versioned): the files with used_in_build=True in
    drive_manifest.csv, by file id (zips by range requests with fetch_drive.py, small files as plain downloads)
"""
import csv, json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

DL, OUT = (os.path.abspath(a) for a in sys.argv[1:3])
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
UA = {"User-Agent": "paiec-ingest/1.0"}
HF_REPO, HF_REV = "xlangai/ubuntu_osworld_verified_trajs", "5473c39e42a538a187a9b2c2b499db59d560fd8c"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REV}/"
SITE_URL = ("https://raw.githubusercontent.com/OS-World/OS-World.github.io/62f8466dbe8b4d67c104b5ead3f8271da01aa09d/"
            "static/data/osworld_verified_results.xlsx")
GIT_URL, HEAD_SHA = "https://github.com/xlang-ai/OSWorld.git", "9f4b4d34319589f3ba6947da6045b0955b08900e"
DRIVE_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
HF_UNUSED = {"qwen36-cua-nothink_300steps_0510.zip", "qwen36-cua-think_300steps_0510.zip",   # OSWorld-V2 / 2.0 tasks
             "results_opus5_task220297_osworld2.0_shareable.zip",
             "results_full361_adaptive_20260609_094019_minimax_m3.zip"}                     # byte-identical duplicate
HOLO = "OSWorldSurfer-Holo3-35B-A3B-20260330-OSWorld-Verified_hcompany-internal-runs-20260416_complete-with-rewards.zip"
HOLO_T = ["072452", "072955", "073458"]
COACT, COACT_TAR = "coact1-150-100-50-15steps.zip", "coact1_traj_150_100_50_15.tar.gz"
SONNET = "oswv_claude-sonnet-4-6.tar.gz"
RX = (r"(^|/)(result\.txt|args\.json|all_result\.json|results\.json|manifest\.json|README\.md|status\.json|final_summary\.json"
      r"|merged_summary_with_repair\.json|phase_results\.json|test_[^/]*\.json|SHA256SUMS|result\.json|LICENSE)$")
RX_HOLO = r"(^|/)task_summary\.json$"
G = os.path.join(DL, "repo", "osworld_full.git")


def run(*a, tries=4):
    for k in range(1, tries):
        if subprocess.run([PY, "-I", *a]).returncode == 0:
            return
        print(f"retry {k} in {60 * k}s: {os.path.basename(a[0])} {a[1] if len(a) > 1 else ''}", flush=True)
        time.sleep(60 * k)                        # Drive and HF errors (HTTP 500, dropped connections) are usually brief
    if subprocess.run([PY, "-I", *a]).returncode:
        raise SystemExit(f"failed: {a}")


def get(url, path, magic=None):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        data = r.read()
    if data.lstrip()[:1] == b"<" or (magic is not None and not data.startswith(magic)):
        raise SystemExit(f"unexpected content from {url}: {data[:60]!r}")
    open(path + ".part", "wb").write(data)
    os.replace(path + ".part", path)


def git(*a, inp=None, lazy=True):
    env = dict(os.environ) if lazy else dict(os.environ, GIT_NO_LAZY_FETCH="1")
    return subprocess.run(["git", "--git-dir", G, *a], input=inp, capture_output=True, check=True, env=env).stdout


def repo():
    if not os.path.exists(G):
        subprocess.run(["git", "init", "-q", "--bare", G], check=True)
        git("remote", "add", "origin", GIT_URL)
        for k, v in (("core.repositoryformatversion", "1"), ("extensions.partialClone", "origin"),
                     ("remote.origin.promisor", "true"), ("remote.origin.partialclonefilter", "blob:none")):
            git("config", k, v)
    if subprocess.run(["git", "--git-dir", G, "cat-file", "-e", HEAD_SHA + "^{commit}"], capture_output=True,
                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1")).returncode:
        git("fetch", "-q", "--no-tags", "--filter=blob:none", "origin", HEAD_SHA)
    oids = set()
    for c in git("rev-list", "--first-parent", HEAD_SHA).decode().split():
        for ln in git("ls-tree", "-r", c, "--", "evaluation_examples/examples", "evaluation_examples/test_all.json",
                      "evaluation_examples/test_nogdrive.json").decode().splitlines():
            meta, p = ln.split("\t", 1)
            if meta.split()[1] == "blob" and p.endswith(".json"):
                oids.add(meta.split()[2])
    chk = git("cat-file", "--batch-check", inp=("\n".join(sorted(oids)) + "\n").encode(), lazy=False).decode().splitlines()
    miss = [ln.split()[0] for ln in chk if ln.endswith(" missing")]
    if miss:
        git("-c", "fetch.negotiationAlgorithm=noop", "fetch", "-q", "--no-tags", "--no-write-fetch-head", "--recurse-submodules=no",
            "--filter=blob:none", "--stdin", "origin", inp=("\n".join(miss) + "\n").encode())
    print(f"repo: {len(oids)} task-file blobs, {len(miss)} fetched", flush=True)


def hf():
    tree = os.path.join(DL, "tree_verified.json")
    if not os.path.exists(tree):
        from huggingface_hub import HfApi
        from huggingface_hub.hf_api import RepoFile
        t = HfApi().list_repo_tree(HF_REPO, repo_type="dataset", revision=HF_REV, recursive=True)
        json.dump([dict(type="file" if isinstance(x, RepoFile) else "directory", path=x.path, size=getattr(x, "size", None))
                   for x in t], open(tree + ".part", "w"))
        os.replace(tree + ".part", tree)
    zips = [x["path"] for x in json.load(open(tree)) if x["type"] == "file" and x["path"].endswith(".zip") and x["path"] not in HF_UNUSED]
    run(f"{HERE}/fetch_hf_list.py", DL, *zips)
    jobs = [lambda: run(f"{HERE}/fetch_hf_members.py", DL, *[z for z in zips if z not in (HOLO, COACT)])]
    d = f"{DL}/members/{SONNET}"
    if not os.path.exists(f"{d}/tar_index.json"):
        jobs.append(lambda: run(f"{HERE}/fetch_stream_tar.py", "targz", HF_URL + SONNET, d, RX))
    d2 = f"{DL}/members/{COACT}"
    if not os.path.exists(f"{d2}/tar_index.json"):
        jobs.append(lambda: run(f"{HERE}/fetch_stream_tar.py", "zipmember", HF_URL + COACT, f"{DL}/listings/{COACT}.json", COACT_TAR, d2, RX))

    def holo():
        for t in HOLO_T:
            lst = f"{DL}/listings/holo3_internal_{t}.json"
            if not os.path.exists(lst):
                run(f"{HERE}/fetch_remote_zip.py", "listnested", HF_URL + HOLO, f"{DL}/listings/{HOLO}.json",
                    f"{HOLO[:-4]}/h_company_internal_runs/trajectories_results_20260416_{t}.zip", lst)
            run(f"{HERE}/fetch_remote_zip.py", "fetch", HF_URL + HOLO, lst, RX_HOLO, f"{DL}/members/holo3_internal_{t}", "12")
    jobs.append(holo)
    with ThreadPoolExecutor(len(jobs)) as ex:
        for f in [ex.submit(j) for j in jobs]:
            f.result()


def drive():
    used = {r["drive_id"]: r for r in csv.DictReader(open(f"{HERE}/drive_manifest.csv")) if r["used_in_build"] == "True"}
    os.makedirs(f"{DL}/drive/small", exist_ok=True)
    ids = f"{DL}/drive/ids_build.txt"
    open(ids, "w").writelines(l.strip() + "\n" for l in open(f"{HERE}/drive_ids.txt")
                              if l.split(" ", 1)[0] in used and used[l.split(" ", 1)[0]]["file_name"].endswith(".zip"))
    run(f"{HERE}/fetch_drive.py", DL, ids)
    for i, r in used.items():
        if not r["file_name"].endswith(".zip"):
            get(DRIVE_URL.format(i), f"{DL}/drive/small/{i}.bin", b"\x1f\x8b" if r["file_name"].endswith(".gz") else None)


os.makedirs(DL, exist_ok=True)
get(SITE_URL, f"{DL}/site/osworld_verified_results.xlsx", b"PK")
with ThreadPoolExecutor(3) as ex:
    for f in [ex.submit(j) for j in (repo, hf, drive)]:
        f.result()
subprocess.run([PY, "-I", f"{HERE}/build.py", DL, OUT], check=True)
