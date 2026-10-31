"""Snapshot the LiveCodeBench Pro leaderboard backend (https://webhook.cp-bench.orzzh.com). Optional refresh tool:
the backend is not versioned, and fetch.py builds from the shipped api_snapshot.json.gz.
Usage: python -I fetch_api.py <out_dir>
Writes llm.json, difficulty/<mode>_<ps>_<diff>.json, submissions/<model>__<provider>__<mode>__<ps>__<diff>.json, fetch_log.json,
then packs them into <out_dir>/api_snapshot.json.gz, the first argument of build.py.
Standard library only."""
import glob, gzip, json, os, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
B = "https://webhook.cp-bench.orzzh.com"
out = sys.argv[1]
os.makedirs(f"{out}/submissions", exist_ok=True); os.makedirs(f"{out}/difficulty", exist_ok=True)
UA = {"User-Agent": "paiec-ingest/1.0 (research snapshot)"}
def get(path, params=None):
    url = B + path + ("?" + urllib.parse.urlencode(params) if params else "")
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (400, 404, 422):
                return e.code, e.read()
            time.sleep(3 * (k + 1))
        except Exception:
            time.sleep(3 * (k + 1))
    return -1, b""
log = {"retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "requests": []}
st, body = get("/leaderboard/llm")
open(f"{out}/llm.json", "wb").write(body)
llms = json.loads(body)
modes = [("static", ps) for ps in ["quater_2024_10_12", "quater_2025_1_3", "quater_2025_4_6", "quater_2025_7_9"]] + [("live", None)]
diffs = ["easy", "medium", "hard"]
for mode, ps in modes:
    for d in diffs:
        p = {"difficulty": d, "benchmark_mode": mode}
        if ps: p["static_problemset"] = ps
        st, body = get("/leaderboard/llm/difficulty", p)
        if st == 200:
            open(f"{out}/difficulty/{mode}_{ps}_{d}.json", "wb").write(body)
        log["requests"].append(["difficulty", p, st, len(body)])
jobs = []
for m in llms:
    for mode, ps in modes:
        for d in diffs:
            jobs.append((m["name"], m["provider"], mode, ps, d))
    jobs.append((m["name"], m["provider"], "rating", None, "all"))
def job(j):
    name, prov, mode, ps, d = j
    fn = f"{out}/submissions/" + urllib.parse.quote(f"{name}__{prov}__{mode}__{ps}__{d}", safe="") + ".json"
    if os.path.exists(fn):
        return j, "cached", 0
    p = {"model_name": name, "model_provider": prov, "difficulty": d, "benchmark_mode": mode}
    if ps: p["problemset_name"] = ps
    st, body = get("/leaderboard/llm/submissions", p)
    if st == 200:
        open(fn, "wb").write(body)
    return j, st, len(body)
with ThreadPoolExecutor(4) as ex:
    for j, st, n in ex.map(job, jobs):
        log["requests"].append(["submissions", j, st, n])
json.dump(log, open(f"{out}/fetch_log.json", "w"))
print("done", len(jobs), "jobs;", sum(1 for r in log["requests"] if r[2] == 200), "ok")
name = lambda f: urllib.parse.unquote(os.path.basename(f)[:-len(".json")])
snap = {"llm": llms, "fetch_log": log,
        "difficulty": {name(f): json.load(open(f)) for f in sorted(glob.glob(f"{out}/difficulty/*.json"))},
        "submissions": {name(f): json.load(open(f)) for f in sorted(glob.glob(f"{out}/submissions/*.json"))}}
open(f"{out}/api_snapshot.json.gz", "wb").write(gzip.compress(json.dumps(snap).encode()))
print("packed", len(snap["difficulty"]), "difficulty and", len(snap["submissions"]), "submission views")
