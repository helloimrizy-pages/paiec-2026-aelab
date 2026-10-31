"""Build dist/<version>_private.zip (OpenAI + Anthropic keys, never share) and dist/<version>_public.zip (no keys).
Includes the public-results library (publib_data/). Usage: python tools/build_zip.py v9_lib
Keys: the environment variables OPENAI_API_KEY, ANTHROPIC_API_KEY and (optional) ANTHROPIC_WORKSPACE_ID, or else
~/.config/paiec/openai.env (OPENAI_API_KEY) and ~/.config/paiec/anthropic.env (ANTHROPIC_API_KEY, ANTHROPIC_WORKSPACE_ID)."""
import json, os, sys, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
version = sys.argv[1] if len(sys.argv) > 1 else "v9_lib"
SRC = ROOT / "submissions" / version
FILES = ["model.py", "judge_prompt.py", "publib.py", "h22_mix.py"]


def env(path):
    return dict(l.split("=", 1) for l in path.read_text().splitlines() if "=" in l and not l.startswith("#")) if path.exists() else {}


def build(name, openai_key, anth_key, anth_ws):
    cfg = json.loads((SRC / "judge_config.json").read_text())
    cfg.update(api_key=openai_key, anthropic_api_key=anth_key, anthropic_workspace_id=anth_ws)
    out = ROOT / "dist" / name
    out.parent.mkdir(exist_ok=True)                                     # dist/ is git-ignored, so a fresh clone lacks it
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(SRC / f, f)
        for f in sorted((SRC / "publib_data").iterdir()):
            z.write(f, f"publib_data/{f.name}")
        z.writestr("judge_config.json", json.dumps(cfg, indent=2))
    out.chmod(0o600)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.2f} MB)")


def verify(openai_key, anth_key, anth_ws):
    """Refuse to build a private zip whose keys the live APIs reject (free model-list calls; keys never printed)."""
    import urllib.error, urllib.request
    def ok(url, headers):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
                return r.status == 200, r.status
        except urllib.error.HTTPError as e:
            return False, e.code
    h = {"x-api-key": anth_key, "anthropic-version": "2023-06-01"}
    if anth_ws:
        h["anthropic-workspace-id"] = anth_ws
    for name, (good, code) in [("OpenAI", ok("https://api.openai.com/v1/models", {"Authorization": f"Bearer {openai_key}"})),
                               ("Anthropic", ok("https://api.anthropic.com/v1/models", h))]:
        print(f"{name} key check: HTTP {code}")
        if not good:
            raise SystemExit(f"{name} credentials rejected (HTTP {code}); check the keys before building.")


cfgdir = Path.home() / ".config" / "paiec"
o, a = env(cfgdir / "openai.env"), env(cfgdir / "anthropic.env")
key = lambda name, f: (os.environ.get(name) or f.get(name, "")).strip()
openai_key, anth_key, anth_ws = key("OPENAI_API_KEY", o), key("ANTHROPIC_API_KEY", a), key("ANTHROPIC_WORKSPACE_ID", a)
if not (openai_key and anth_key):
    raise SystemExit("set OPENAI_API_KEY and ANTHROPIC_API_KEY (environment or ~/.config/paiec/*.env)")
verify(openai_key, anth_key, anth_ws)
build(f"{version}_private.zip", openai_key, anth_key, anth_ws)
build(f"{version}_public.zip", "", "", "")
