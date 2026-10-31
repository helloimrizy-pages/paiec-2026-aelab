"""Download the pinned Multi-LCB sources, then run build.py.

Usage: python -I fetch.py <download_dir> <out_dir>

Sources:
  https://github.com/Multi-LCB/multi-lcb.github.io @ e1f940ea1a767d46ff6ef09411cafa3917524c8a
      sparse checkout: data/lcb/index.json, and meta.json + <language>.jsonl of every listed model
      whose name does not start with _old or _del
  https://github.com/Multi-LCB/Multi-LCB @ d80be9f24268e9e5c6e1463e26c9f05417fe06a0 (branch release)
      the five prompt files build.py reads (raw.githubusercontent.com)
  https://huggingface.co/datasets/livecodebench/code_generation_lite @ 0fe84c3912ea0c4d4a78037083943e8f0c4dd505
      test.jsonl..test6.jsonl (about 4.5 GB) are streamed; only the statement fields are written to
      <file>.slim.jsonl, the private test cases are not stored.
Files already present are skipped, so a rerun resumes.
"""
import hashlib, json, os, re, subprocess, sys, urllib.request

SITE_URL = "https://github.com/Multi-LCB/multi-lcb.github.io"
SITE_SHA = "e1f940ea1a767d46ff6ef09411cafa3917524c8a"
CODE_SHA = "d80be9f24268e9e5c6e1463e26c9f05417fe06a0"
REV = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
KEEP = ["question_title", "question_content", "platform", "question_id", "contest_id", "contest_date",
        "starter_code", "difficulty", "metadata", "public_test_cases"]
CODE_FILES = ["lcb_runner/prompts/code_generation.py", "lcb_runner/runner/parser.py", "prompt_enhancer/enhancer.py",
              "prompt_enhancer/example_parser.py", "prompt_enhancer/unified_converter.py"]
# build.py imports these two pure-stdlib modules by path; their content is pinned here
SHA256 = {"prompt_enhancer/example_parser.py": "a80197fd9877884fab0bf991232b94d23122c6ff69658adb05a496bbf018ef2d",
          "prompt_enhancer/unified_converter.py": "f0e236acbf23a8ec5b690750c29ae9f8acdd049d82dd8c1cedeac06c14a59e4b"}
HERE = os.path.dirname(os.path.abspath(__file__))


def git(d, *a, **kw):
    return subprocess.run(["git", "-C", d, *a], check=True, **kw)


def fetch_site(d):
    if not os.path.isdir(f"{d}/.git"):
        os.makedirs(d, exist_ok=True)
        git(d, "init", "-q")
        git(d, "remote", "add", "origin", SITE_URL)
        git(d, "config", "core.sparseCheckout", "true")
    if subprocess.run(["git", "-C", d, "cat-file", "-e", f"{SITE_SHA}^{{commit}}"], capture_output=True).returncode:
        git(d, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", SITE_SHA)
    index = json.loads(git(d, "show", f"{SITE_SHA}:data/lcb/index.json", capture_output=True).stdout)
    paths = ["data/lcb/index.json"]
    for m in index["models"]:
        if not m["name"].startswith(("_old", "_del")):
            paths += [f"data/lcb/{m['name']}/meta.json"] + [f"data/lcb/{m['name']}/{f}" for f in m["files"].values()]
    with open(f"{d}/.git/info/sparse-checkout", "w") as f:
        f.writelines("/" + re.sub(r"([*?\[\\])", r"\\\1", p) + "\n" for p in paths)
    git(d, "checkout", "-q", "--detach", SITE_SHA)
    missing = [p for p in paths if not os.path.isfile(f"{d}/{p}")]
    assert not missing, missing[:5]
    print("site:", len(paths), "files")


def fetch_code(d):
    for p in CODE_FILES:
        dst = f"{d}/{p}"
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            url = f"https://raw.githubusercontent.com/Multi-LCB/Multi-LCB/{CODE_SHA}/{p}"
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            with open(dst + ".part", "wb") as f:
                f.write(data)
            os.replace(dst + ".part", dst)
        if p in SHA256:
            assert hashlib.sha256(open(dst, "rb").read()).hexdigest() == SHA256[p], p
    print("code:", len(CODE_FILES), "files")


def fetch_lcb(d):
    os.makedirs(d, exist_ok=True)
    for f in LCB_FILES:
        dst = f"{d}/{f.replace('.jsonl', '.slim.jsonl')}"
        if os.path.exists(dst):
            continue
        url = f"https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/{REV}/{f}"
        n = 0
        with urllib.request.urlopen(url, timeout=120) as r, open(dst + ".part", "w") as out:
            for line in r:
                if not line.strip():
                    continue
                q = json.loads(line)
                s = {k: q.get(k) for k in KEEP}
                s["source_file"] = f
                out.write(json.dumps(s, ensure_ascii=False) + "\n")
                n += 1
        os.replace(dst + ".part", dst)
        print(f, n)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    dl, out = (os.path.abspath(a) for a in sys.argv[1:3])
    os.makedirs(out, exist_ok=True)
    fetch_site(f"{dl}/site")
    fetch_code(f"{dl}/code")
    fetch_lcb(f"{dl}/lcb")
    subprocess.run([sys.executable, "-I", f"{HERE}/build.py", f"{dl}/site", f"{dl}/code", f"{dl}/lcb", out],
                   check=True, cwd=out)
