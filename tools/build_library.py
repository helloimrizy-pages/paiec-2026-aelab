"""Download the public per-item results of the library benchmarks and rebuild the library index.

  python tools/build_library.py                 # every benchmark in data/public_results/READY.txt, then the index
  python tools/build_library.py cybench osworld # only these benchmarks, then the index
  python tools/build_library.py --export        # also write the index into submissions/v9_lib/publib_data/

Per benchmark, data/public_results/<slug>/fetch.py downloads the pinned sources into data/downloads/<slug>/ and writes
items.parquet / results.parquet next to itself. lab/h22_library.py then builds the index into lab/features/h22_lib/.
Model names are mapped with map_model_registry.json from aims-foundations/measurement-db at a pinned commit.
"""
import hashlib, subprocess, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "public_results"
REG = "scripts/build_measurement_tables/map_model_registry.json"
REG_URL = f"https://raw.githubusercontent.com/aims-foundations/measurement-db/652474f73fed42ebfe08690a7d48e87a78b78dd4/{REG}"
REG_SHA256 = "0d6c408a422acabf5ca4309418d05e47ef482e6ea6492d40aaa53af76bdf8fee"


def registry():
    p = ROOT / "measurement-db-code" / REG
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(REG_URL, timeout=60) as r:
            p.write_bytes(r.read())
    if hashlib.sha256(p.read_bytes()).hexdigest() != REG_SHA256:
        raise SystemExit(f"{p}: unexpected content (expected sha256 {REG_SHA256})")


def run(*args, cwd=ROOT):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, "-I", *map(str, args)], cwd=cwd, check=True)


args = [a for a in sys.argv[1:] if not a.startswith("--")]
ready = [l.strip() for l in open(SRC / "READY.txt") if l.strip() and not l.startswith("#")]
for slug in args or ready:
    if slug not in ready:
        raise SystemExit(f"{slug}: not in READY.txt")
    run(SRC / slug / "fetch.py", ROOT / "data" / "downloads" / slug, SRC / slug)
registry()
run("h22_library.py", "build", cwd=ROOT / "lab")
if "--export" in sys.argv:
    run("h22_library.py", "export", ROOT / "submissions" / "v9_lib" / "publib_data", cwd=ROOT / "lab")
