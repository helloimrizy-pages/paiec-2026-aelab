"""Download the Terminal-Bench sources, then run build.py.

Usage: python -I fetch.py <download_dir> <out_dir> [--live]

Sources
- Harbor Hub public read API (https://ofhuhcpkvzjlejydnvyd.supabase.co, not versioned): fetch_hub.py writes
  <download_dir>/hub/*.json. By default the build uses hub_snapshot.tar.xz, the Hub data retrieved 2026-10-21T16:02:05Z
  (the data behind the submitted library). --live fetches the current Hub instead and reports what changed since the snapshot.
- HF dataset harborframework/terminal-bench-2-leaderboard @ 572b2614be2c0cb2527e14f5b1e4026f1072e6c1:
  submissions/terminal-bench/2.0/<submission>/metadata.yaml and <job>/<trial>/result.json (sparse checkout).
- GitHub harbor-framework/terminal-bench @ 2e29ca91d5f1cb6fb60924bf867acc755565c044: leaderboard/submissions/*.json (TB 4.0);
  tags v3.0.0 = 2b0442c3c583b710ca8da14c8e601b99f2f1f244 and v4.0.0 = 452bf305c6daa62fc59061d22133a7cbc7c1572e: tasks/*/instruction.md.
- GitHub harbor-framework/terminal-bench-2-1 @ 7131e4375048a0e408a8fb404b5f499d726b695b: leaderboard/submissions/*.json
  and tasks/*/instruction.md.
- GitHub harbor-framework/terminal-bench-2 @ 2fd12b88aafdd04a52c298e3940bcb189f9766d6: */instruction.md.
Git sources are shallow, blob-less, sparse fetches of the pinned commit. Finished steps are skipped on a rerun.
"""
import glob, json, os, shutil, subprocess, sys, tarfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
DL, OUT = (os.path.abspath(p) for p in sys.argv[1:3])
GH = "https://github.com/harbor-framework/"
REPOS = {  # dest under <download_dir>: (url, commit, sparse patterns)
    "hf_tb2_git": ("https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard",
                   "572b2614be2c0cb2527e14f5b1e4026f1072e6c1",
                   ["/submissions/terminal-bench/2.0/*/metadata.yaml", "/submissions/terminal-bench/2.0/*/*/*/result.json"]),
    "git/tb_head": (GH + "terminal-bench.git", "2e29ca91d5f1cb6fb60924bf867acc755565c044", ["/leaderboard/submissions/*.json"]),
    "git/export/tb_v4.0.0": (GH + "terminal-bench.git", "452bf305c6daa62fc59061d22133a7cbc7c1572e", ["/tasks/*/instruction.md"]),
    "git/export/tb_v3.0.0": (GH + "terminal-bench.git", "2b0442c3c583b710ca8da14c8e601b99f2f1f244", ["/tasks/*/instruction.md"]),
    "git/export/tb21_head": (GH + "terminal-bench-2-1.git", "7131e4375048a0e408a8fb404b5f499d726b695b",
                             ["/tasks/*/instruction.md", "/leaderboard/submissions/*.json"]),
    "git/export/tb2_head": (GH + "terminal-bench-2.git", "2fd12b88aafdd04a52c298e3940bcb189f9766d6", ["/*/instruction.md"]),
}
COPY = {"raw_lb/submissions": "git/tb_head/leaderboard/submissions", "raw_tb21/submissions": "git/export/tb21_head/leaderboard/submissions"}
ENV = os.environ | {"GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
SNAP, HUB = f"{HERE}/hub_snapshot.tar.xz", ("boards.json", "trials.json", "task_versions.json")


def git(d, *a):
    return subprocess.run(["git", "-C", d, *a], check=True, env=ENV, capture_output=True, text=True).stdout.strip()


def sparse_clone(dest, url, sha, patterns):
    if os.path.isdir(f"{dest}/.git") and subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True,
                                                        text=True).stdout.strip() == sha:
        return
    os.makedirs(dest, exist_ok=True)
    if not os.path.isdir(f"{dest}/.git"):
        git(dest, "init", "-q"); git(dest, "remote", "add", "origin", url)
    git(dest, "sparse-checkout", "set", "--no-cone", *patterns)
    git(dest, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha)
    git(dest, "checkout", "-q", "--detach", sha)
    assert git(dest, "rev-parse", "HEAD") == sha, dest


def hub_state(load):  # the Hub records build.py reads, keyed by id
    b, t, tv = (load(n) for n in HUB)
    return ({r["id"]: r for x in b["boards"].values() for r in x["rows"]},
            {v: x["dataset_version_tasks"] for v, x in b["boards"].items()},
            {x["id"]: x for x in t}, {x["id"]: x for x in tv}), b["retrieved_at_utc"]


t0 = time.time()
with tarfile.open(SNAP) as tf:
    if "--live" not in sys.argv[3:]:
        os.makedirs(f"{DL}/hub", exist_ok=True)
        for n in HUB:
            open(f"{DL}/hub/{n}", "wb").write(tf.extractfile(n).read())
    else:
        if not all(os.path.exists(f"{DL}/hub/{n}") for n in HUB):
            subprocess.run([sys.executable, "-I", f"{HERE}/fetch_hub.py", f"{DL}/hub"], check=True)
        (old, when), (new, now) = hub_state(lambda n: json.load(tf.extractfile(n))), hub_state(lambda n: json.load(open(f"{DL}/hub/{n}")))
        diff = {name: sum(a.get(k) != b.get(k) for k in a.keys() | b.keys())
                for name, a, b in zip(("board rows", "dataset task lists", "trials", "task versions"), old, new)}
        print(f"live Hub ({now}) vs hub_snapshot.tar.xz ({when}): records added, removed or changed: {diff}", flush=True)
        if any(diff.values()):
            print("WARNING: the Hub changed since the snapshot; outputs differ from the submitted library. "
                  "Rerun without --live to build from the snapshot.", flush=True)
print(f"hub done {time.time() - t0:.0f} s", flush=True)
for dest, (url, sha, pats) in REPOS.items():
    sparse_clone(f"{DL}/{dest}", url, sha, pats)
    print(f"{dest} @ {sha[:8]} done {time.time() - t0:.0f} s", flush=True)
for dest, src in COPY.items():
    if not os.path.isdir(f"{DL}/{dest}"):
        os.makedirs(f"{DL}/{dest}.part", exist_ok=True)
        for p in glob.glob(f"{DL}/{src}/*.json"):
            shutil.copy(p, f"{DL}/{dest}.part/")
        os.replace(f"{DL}/{dest}.part", f"{DL}/{dest}")
subprocess.run([sys.executable, "-I", f"{HERE}/build.py", DL, OUT], check=True)
