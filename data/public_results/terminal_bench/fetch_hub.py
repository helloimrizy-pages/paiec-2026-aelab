"""Fetch Terminal-Bench leaderboard rows, their trials, and task instructions from the public
Harbor Hub (Supabase PostgREST, anonymous publishable key from the harbor 0.24.0 wheel, harbor/auth/constants.py).
The Hub is not versioned. Writes boards.json, trials.json, task_versions.json and jobs.json.
Usage: python -I fetch_hub.py <out_dir>"""
import json, sys, os, time, urllib.request, urllib.parse
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
U = "https://ofhuhcpkvzjlejydnvyd.supabase.co"; K = "sb_publishable_Z-vuQbpvpG-PStjbh4yE0Q_e-d3MTIH"
H = {"apikey": K, "Content-Type": "application/json"}
BOARDS = {"2.0": "f58e955f-4db5-48a0-87ec-1c324a4a9414", "2.1": "60330f75-0dd8-47ea-bd1d-e2ea28945731",
          "3.0": "694bb3b0-c5d5-4899-93de-4c392b814d7e", "4.0": "9f966760-00f1-424e-90f5-c964fb6f6091"}

def req(path, body=None, tries=5):
    for i in range(tries):
        try:
            r = urllib.request.Request(U + path, data=None if body is None else json.dumps(body).encode(), headers=H,
                                       method="GET" if body is None else "POST")
            with urllib.request.urlopen(r, timeout=120) as f:
                return json.loads(f.read())
        except Exception as e:
            print("retry", i, path[:120], e, flush=True); time.sleep(2 + 3 * i)
    raise RuntimeError(path)

def rest(table, params):
    return req(f"/rest/v1/{table}?" + urllib.parse.urlencode(params, safe="(),.*:"))

def chunks(xs, n):
    for i in range(0, len(xs), n): yield xs[i:i + n]

snap = {"retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "boards": {}}
all_trial_ids = []
for ver, lb_id in BOARDS.items():
    d = req("/functions/v1/leaderboard-read", {"leaderboard_id": lb_id, "page": 1, "page_size": 500})
    assert d["pagination"]["total_pages"] == 1, d["pagination"]
    rows = d["rows"]
    for r in rows:
        tids = []
        start = 0
        while True:
            page = rest("leaderboard_row_trial", {"select": "trial_id", "row_id": f"eq.{r['id']}",
                                                   "order": "trial_id", "offset": start, "limit": 1000})
            tids += [p["trial_id"] for p in page]
            if len(page) < 1000: break
            start += 1000
        r["trial_ids"] = tids; all_trial_ids += tids
    dvs = rest("dataset_version", {"select": "*", "id": "in.(" + ",".join(d["leaderboard"]["dataset_version_ids"]) + ")"})
    dv_tasks = {}
    for dv in dvs:
        links = []; start = 0
        while True:
            page = rest("dataset_version_task", {"select": "task_version_id,task_version:task_version_id(content_hash,package:package_id(name,org:org_id(name)))",
                                                  "dataset_version_id": f"eq.{dv['id']}", "order": "task_version_id",
                                                  "offset": start, "limit": 1000})
            links += page
            if len(page) < 1000: break
            start += 1000
        dv_tasks[dv["id"]] = links
    snap["boards"][ver] = {"leaderboard": d["leaderboard"], "rows": rows, "dataset_versions": dvs, "dataset_version_tasks": dv_tasks}
    print(ver, "rows", len(rows), "trials", sum(len(r["trial_ids"]) for r in rows), "dataset versions", len(dvs),
          {k: len(v) for k, v in dv_tasks.items()}, flush=True)

cols = ("id,job_id,trial_name,task_name,task_content_hash,rewards,exception_type,status,error_disposition,"
        "config,started_at,finished_at,created_at,hosted_error")
trials = []
for c in chunks(sorted(set(all_trial_ids)), 80):
    trials += rest("trial", {"select": cols, "id": "in.(" + ",".join(c) + ")"})
print("trial records", len(trials), "of", len(set(all_trial_ids)), flush=True)
hashes = sorted({t["task_content_hash"] for t in trials if t.get("task_content_hash")})
for b in snap["boards"].values():
    for links in b["dataset_version_tasks"].values():
        hashes += [l["task_version"]["content_hash"] for l in links if l.get("task_version")]
hashes = sorted(set(hashes))
tvs = []
for c in chunks(hashes, 40):
    tvs += rest("task_version", {"select": "id,content_hash,instruction,description,metadata,published_at,yanked_at,archive_path,package:package_id(name,org:org_id(name))",
                                 "content_hash": "in.(" + ",".join(c) + ")"})
print("task versions", len(tvs), "of", len(hashes), flush=True)
jobs = sorted({t["job_id"] for t in trials})
jobrows = []
for c in chunks(jobs, 80):
    jobrows += rest("job", {"select": "id,job_name,created_at,config", "id": "in.(" + ",".join(c) + ")"})
print("jobs", len(jobrows), "of", len(jobs), flush=True)
for name, obj in (("boards", snap), ("trials", trials), ("task_versions", tvs), ("jobs", jobrows)):
    with open(f"{OUT}/{name}.json.part", "w") as f:
        json.dump(obj, f)
    os.replace(f"{OUT}/{name}.json.part", f"{OUT}/{name}.json")
