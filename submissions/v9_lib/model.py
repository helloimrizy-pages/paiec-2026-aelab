"""PAIEC v7: v5 + item-metadata groups + adaptive answer-length slope + same-item sibling evidence.

v7 additions (all label-driven, no extra inference requests):
  - answer a = -log(1 + answer tokens) of the same attempt; standardized deviation (a_i - abar)/ANSWER_SCALE enters
    the label likelihood with a zero-mean random slope x ~ N(0, SA^2): it only matters when labels support it.
  - groups: key=value tokens of item_features (keys containing "idx"/ending in "id" and prompt_source dropped; tokens
    shared by every visible item dropped). With posterior predictive p_l of the revealed items, residual r = y - p_l,
    each token t gets delta_t = sum r / (sum p(1-p) + n_tok/GROUP_SD^2); the target logit moves by sum of its deltas.
  - sibling evidence: if another subject's revealed label on this benchmark is on the same item, the target logit moves
    by SIB_GAMMA * mean(y_sib - rate_sib), rate_sib = (sum of that subject's labels here + 1)/(n + 2).


Judge. An OpenAI model (judge_config.json, "model"/"effort") reads the subject and item fields and returns
p_success, typical_success_rate and relative_difficulty (judge_prompt.py; the anonymous benchmark_id is not
shown). z = logit(p_success), t = logit(typical_success_rate), rel in {-2..2}.

Effort. The same model, at higher reasoning effort, attempts the item exactly as written (no answer key, no
environment). e = -log(1 + total output tokens, including reasoning and answer).
Effort is subject-independent, so one attempt per distinct item is stored and reused.

Budget 0 (no labels of this subject on this benchmark): p = sigmoid(HEAD_C + HEAD_W . [z, rel, t]).
Effort is not used here: across benchmarks long reasoning goes with high success (mathematics), within a
benchmark with low success, so it is only informative relative to the pair's other items.

Labels present: with zbar, ebar the means over the items this call can see for the pair (the target plus the
revealed items for the target; the revealed items for the revealed items),
    m_i = MUND_C + MUND_BB * zbar + MUND_BW * (z_i - zbar) + MUND_BE * (e_i - ebar),
    logit P(y_i = 1) = m_i + u + v * (z_i - zbar) + w * (e_i - ebar),
    u ~ 0.7 N(0, 0.5^2) + 0.3 N(0, 2^2)   (pair-level correction; heavy tails for large judge errors)
    v ~ N(0, 0.6^2)                        (per-pair correction of the within-pair judge slope)
    w ~ N(0, 0.4^2)                        (independent total-token slope correction)
The prediction integrates the posterior with Gauss-Hermite quadrature. Numerical
accuracy was checked against denser quadrature in the recorded offline experiments.

All constants were fitted on public measurement-db training pairs; design choices were made by
leave-one-benchmark-out simulation of the competition protocol. v8: the budget-0 head and the pair-level mapping
(HEAD_C, HEAD_W, MUND_C, MUND_BB) are the mean of the 4-pilot-benchmark fit (v7) and a fit on 97 training-pool
benchmarks of the measurement-db collection (rev 54074453), which halves the between-pair slope; the within-pair
slopes and everything else are unchanged from v7. Judgments and attempts are label-independent
functions of their input and are stored in judge_cache/ ("cache": true). Every call keeps to a time budget;
an attempt that runs out of time is recorded as high effort (TIMEOUT_TOKENS), and if the judge fails the call
falls back to the constant prior of submission v1, so network faults never fail the run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss

from judge_prompt import INSTRUCTIONS, parse_output_text, request_body, user_message
from publib import PubLib, harness_match
from h22_mix import priors as lib_priors, mix_predict, easy_predict

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "judge_config.json").read_text())
MODEL = CFG.get("model", "gpt-5.6-luna")
EFFORT = CFG.get("effort", "low")
HEAD_C, HEAD_W = float(CFG["head_c"]), np.asarray(CFG["head_w"], dtype=float)
MUND_C, MUND_BB, MUND_BW, MUND_BE = (float(CFG[k]) for k in ("mund_c", "mund_bB", "mund_bW", "mund_bE"))
ATTEMPT_EFFORT = CFG.get("attempt_effort", "medium")
ATTEMPT_MAX_TOKENS = int(CFG.get("attempt_max_output_tokens", 25000))
ATTEMPT_TIMEOUT = float(CFG.get("attempt_timeout", 100))
TIMEOUT_TOKENS = float(CFG.get("timeout_tokens", 8000))
CALL_BUDGET = float(CFG.get("call_budget", 110))
USE_CACHE = bool(CFG.get("cache", True))
API_KEY = os.environ.get("OPENAI_API_KEY") or CFG.get("api_key", "")
BASE_URL = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
# Second attempter (v9, H23): Claude Haiku 5.5 through the Anthropic Messages API; effort = mean of both attempters.
ATTEMPT2 = bool(CFG.get("attempt2", True))
ATTEMPT2_MODEL = CFG.get("attempt2_model", "claude-haiku-5-5")
ATTEMPT2_EFFORT = CFG.get("attempt2_effort", "medium")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or CFG.get("anthropic_api_key", "")
ANTHROPIC_WS = os.environ.get("ANTHROPIC_WORKSPACE_ID") or CFG.get("anthropic_workspace_id", "")
# Public-results library (v9, H22).
USE_LIB = bool(CFG.get("use_library", True))
EASY_MODE = CFG.get("easy_mode", "easy")          # item known, subject's model not: "easy" | "base"
try:
    LIB = PubLib(HERE / "publib_data") if USE_LIB and (HERE / "publib_data" / "items.json").exists() else None
except Exception:                                  # a broken library must never fail the run
    LIB = None

MU = -0.573        # fallback constant prior logit (submission v1)
S0, SZ = 0.75, 0.6
SE = float(CFG.get("effort_slope_sd", 0.4))
EPS = 0.01
JITTER = 1e-6
P_MIN, P_MAX = 0.001, 0.999
SA = float(CFG.get("answer_slope_sd", 0.3))
ANSWER_SCALE = float(CFG.get("answer_scale", 0.755531800877751))
GROUP_SD = float(CFG.get("group_sd", 1.0))
SIB_GAMMA = float(CFG.get("sibling_gamma", 1.6))
DROP_BELOW = 25.0
B0_TAU = float(CFG.get("b0_tau", 1.25))
B0_X, B0_W = hermgauss(21)
B0_X, B0_W = np.sqrt(2.0) * B0_X, B0_W / np.sqrt(np.pi)  # posterior nodes more than this many log units below the maximum are dropped
RETRYABLE = {408, 409, 429, 500, 502, 503, 504}
REL = {"much_easier": 2.0, "easier": 1.0, "typical": 0.0, "harder": -1.0, "much_harder": -2.0}

# Independent prior nodes for level, judge slope, total-token slope and answer-token slope.
_nu, _nv, _ne, _na = CFG.get("integration_nodes", [41, 21, 9, 7])
_ux, _uw = hermgauss(int(_nu))
_ux, _uw = np.sqrt(2.0) * _ux, _uw / np.sqrt(np.pi)
_u = np.concatenate([0.5 * _ux, 2.0 * _ux])
_uw = np.concatenate([0.7 * _uw, 0.3 * _uw])
_vx, _vw = hermgauss(int(_nv))
_ex, _ew = hermgauss(int(_ne))
_ax, _aw = hermgauss(int(_na))
_v, _vw = SZ * np.sqrt(2.0) * _vx, _vw / np.sqrt(np.pi)
_e, _ew = SE * np.sqrt(2.0) * _ex, _ew / np.sqrt(np.pi)
_a, _aw = SA * np.sqrt(2.0) * _ax, _aw / np.sqrt(np.pi)
_gu, _gv, _ge, _ga = np.meshgrid(_u, _v, _e, _a, indexing="ij")
GRID_U, GRID_V, GRID_E, GRID_A = _gu.ravel(), _gv.ravel(), _ge.ravel(), _ga.ravel()
GRID_LW = np.log((_uw[:, None, None, None] * _vw[None, :, None, None] * _ew[None, None, :, None] * _aw[None, None, None, :]).ravel())
# Fallback needs only the one-dimensional marginal of the level prior.
FALLBACK_U, FALLBACK_LW = _u, np.log(_uw)

CACHE_DIR = HERE / "judge_cache"
_memory: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Cache with cross-worker locks
# ---------------------------------------------------------------------------

def _cache_get(key: str):
    if key in _memory:
        return _memory[key]
    if USE_CACHE:
        try:
            v = json.loads((CACHE_DIR / f"{key}.json").read_text())
            _memory[key] = v
            return v
        except (OSError, ValueError):
            return None
    return None


def _cache_put(key: str, value: dict) -> None:
    _memory[key] = value
    if not USE_CACHE:
        return
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        tmp = CACHE_DIR / f".{key}.{os.getpid()}.{random.getrandbits(32)}.tmp"
        tmp.write_text(json.dumps(value))
        os.replace(tmp, CACHE_DIR / f"{key}.json")
    except OSError:
        pass


def _acquire(key: str, stale_after: float) -> bool:
    """True if this process should compute `key`; False if another worker is computing it."""
    if not USE_CACHE:
        return True
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        path = CACHE_DIR / f"{key}.lock"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if time.time() - path.stat().st_mtime > stale_after:  # abandoned lock
                path.unlink(missing_ok=True)
                return _acquire(key, stale_after)
            return False
    except OSError:
        return True


def _release(key: str) -> None:
    try:
        (CACHE_DIR / f"{key}.lock").unlink(missing_ok=True)
    except OSError:
        pass


def _cached_compute(key: str, compute, deadline: float, stale_after: float):
    v = _cache_get(key)
    if v is not None:
        return v
    if _acquire(key, stale_after):
        try:
            v = compute()
            if v is not None:
                _cache_put(key, v)
            return v
        finally:
            _release(key)
    while time.monotonic() < deadline:  # another worker is computing it: wait for its result
        time.sleep(0.5)
        v = _cache_get(key)
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# OpenAI Responses API
# ---------------------------------------------------------------------------

def _post(body: dict, deadline: float, per_try: float):
    """POST /responses with retries inside the deadline. Returns (json, timed_out)."""
    if not API_KEY:
        return None, False
    data = json.dumps(body).encode("utf-8")
    attempt = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining < 3:
            return None, True
        req = urllib.request.Request(f"{BASE_URL}/responses", data=data, method="POST",
                                     headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=min(per_try, remaining)) as resp:
                return json.loads(resp.read().decode("utf-8")), False
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE:
                return None, False
        except TimeoutError:
            return None, True
        except (urllib.error.URLError, OSError, ValueError) as e:
            if isinstance(getattr(e, "reason", None), TimeoutError):
                return None, True
        attempt += 1
        pause = min(10.0, 1.5 * 2 ** attempt) * (0.5 + random.random())
        time.sleep(max(0.0, min(pause, deadline - time.monotonic())))


def _output_text(out: dict) -> str:
    return "".join(part.get("text", "") for o in out.get("output", []) if o.get("type") == "message"
                   for part in o.get("content", []) if part.get("type") == "output_text")


def _logit(p: float) -> float:
    p = max(EPS, min(1 - EPS, float(p)))
    return math.log(p / (1 - p))


def _judge(subject: dict, item: dict, deadline: float):
    key = hashlib.sha256(json.dumps(["judge-v4", MODEL, EFFORT, INSTRUCTIONS, user_message(subject, item)],
                                    sort_keys=True).encode("utf-8")).hexdigest()

    def compute():
        out, _ = _post(request_body(subject, item, MODEL, EFFORT), deadline, per_try=90)
        if out is None:
            return None
        try:
            d = parse_output_text(_output_text(out))
        except (ValueError, KeyError):
            return None
        return {"z": _logit(d["p_success"]), "t": _logit(d["typical_success_rate"]),
                "rel": REL.get(d.get("relative_difficulty"), 0.0)}
    return _cached_compute(key, compute, deadline, stale_after=120)


def _attempt(item: dict, deadline: float):
    content = str(item.get("item_content") or "")
    key = hashlib.sha256(json.dumps(["attempt-v5-total", MODEL, ATTEMPT_EFFORT, ATTEMPT_MAX_TOKENS, content]).encode("utf-8")).hexdigest()

    def compute():
        body = {"model": MODEL, "input": content, "reasoning": {"effort": ATTEMPT_EFFORT},
                "max_output_tokens": ATTEMPT_MAX_TOKENS, "store": False}
        out, timed_out = _post(body, deadline, per_try=ATTEMPT_TIMEOUT)
        if out is not None:
            u = out.get("usage") or {}
            try:
                ot = float(u["output_tokens"])
                rt = float((u.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
            except (KeyError, TypeError, ValueError):
                return None
            if not math.isfinite(ot) or not math.isfinite(rt) or ot < rt or rt < 0:
                return None
            return {"ot": ot, "rt": rt}
        if timed_out:
            return {"ot": TIMEOUT_TOKENS, "rt": TIMEOUT_TOKENS, "timeout": True}
        return None
    return _cached_compute(key, compute, deadline, stale_after=ATTEMPT_TIMEOUT + 30)


def _post_anthropic(body: dict, deadline: float, per_try: float):
    """POST /v1/messages with retries inside the deadline. Returns (json, timed_out)."""
    if not ANTHROPIC_KEY:
        return None, False
    data = json.dumps(body).encode("utf-8")
    headers = {"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    if ANTHROPIC_WS:
        headers["anthropic-workspace-id"] = ANTHROPIC_WS
    attempt = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining < 3:
            return None, True
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=data, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=min(per_try, remaining)) as resp:
                return json.loads(resp.read().decode("utf-8")), False
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE and e.code != 529:
                return None, False
        except TimeoutError:
            return None, True
        except (urllib.error.URLError, OSError, ValueError) as e:
            if isinstance(getattr(e, "reason", None), TimeoutError):
                return None, True
        attempt += 1
        pause = min(10.0, 1.5 * 2 ** attempt) * (0.5 + random.random())
        time.sleep(max(0.0, min(pause, deadline - time.monotonic())))


def _attempt2(item: dict, deadline: float):
    """Second self-attempt (Claude Haiku 5.5): only the output-token count is kept."""
    if not ATTEMPT2:
        return None
    content = str(item.get("item_content") or "")
    key = hashlib.sha256(json.dumps(["attempt2", ATTEMPT2_MODEL, ATTEMPT2_EFFORT, ATTEMPT_MAX_TOKENS, content]).encode("utf-8")).hexdigest()

    def compute():
        body = {"model": ATTEMPT2_MODEL, "max_tokens": ATTEMPT_MAX_TOKENS, "output_config": {"effort": ATTEMPT2_EFFORT},
                "messages": [{"role": "user", "content": content or " "}]}
        out, timed_out = _post_anthropic(body, deadline, per_try=ATTEMPT_TIMEOUT)
        if out is not None:
            if out.get("stop_reason") == "refusal":
                return None
            try:
                ot = float((out.get("usage") or {})["output_tokens"])
            except (KeyError, TypeError, ValueError):
                return None
            return {"ot": ot} if math.isfinite(ot) and ot >= 0 else None
        if timed_out:
            return {"ot": TIMEOUT_TOKENS, "timeout": True}
        return None
    return _cached_compute(key, compute, deadline, stale_after=ATTEMPT_TIMEOUT + 30)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _fallback_gp(y: np.ndarray) -> float:
    """Constant-prior Bayesian update with the marginal level prior."""
    lw = FALLBACK_LW.copy()
    if len(y):
        f = MU + FALLBACK_U[None, :] * np.ones((len(y), 1))
        lw = lw + (y[:, None] * f - np.logaddexp(0, f)).sum(0)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    return float(_sigmoid(MU + FALLBACK_U) @ w)


def _tokens(features) -> frozenset:
    if features is None:
        return frozenset()
    out = []
    for part in re.split(r"[;\n]", str(features)):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        if "idx" in k or k.endswith("id") or k == "prompt_source":
            continue
        out.append(f"{k}={v.strip()}")
    return frozenset(out)


def _posterior_predict(m_l, d_l, e_l, a_l, y, m_t, d_t, e_t, a_t, tok_l, tok_t, extra) -> float:
    f = (m_l[:, None] + GRID_U[None, :] + d_l[:, None] * GRID_V[None, :] + e_l[:, None] * GRID_E[None, :]
         + a_l[:, None] * GRID_A[None, :])
    lw = GRID_LW + (y[:, None] * f - np.logaddexp(0, f)).sum(0)
    keep = lw > lw.max() - DROP_BELOW
    w = np.exp(lw[keep] - lw[keep].max())
    w /= w.sum()
    shift = extra
    shared = frozenset.intersection(tok_t, *tok_l)
    own = [t for t in tok_t if t not in shared]
    if own:
        p_l = _sigmoid(f[:, keep]) @ w
        r, h = y - p_l, p_l * (1 - p_l)
        s2 = GROUP_SD ** 2 / len(own)
        for t in own:
            idx = [j for j, tl in enumerate(tok_l) if t in tl]
            if idx:
                shift += float(r[idx].sum()) / (float(h[idx].sum()) + 1.0 / s2)
    return float(_sigmoid(m_t + shift + GRID_U[keep] + d_t * GRID_V[keep] + e_t * GRID_E[keep] + a_t * GRID_A[keep]) @ w)


def _unique(items: list) -> tuple[list, list]:
    keys = [json.dumps([it.get("item_content"), it.get("item_features"), it.get("interactors")]) for it in items]
    first = {}
    for k, it in zip(keys, items):
        first.setdefault(k, it)
    return keys, first


def _sibling_shift(subject, item, bench, labeled) -> float:
    """SIB_GAMMA * mean residual of other subjects' revealed labels on this same item."""
    key = json.dumps([item.get("item_content"), item.get("item_features"), item.get("interactors")])
    per_subject: dict[str, list] = {}
    for (obs_subject, obs_item), label in labeled or []:
        if obs_subject == subject or obs_item.get("benchmark_id", "") != bench:
            continue
        sid = json.dumps(obs_subject, sort_keys=True, default=str)
        same = json.dumps([obs_item.get("item_content"), obs_item.get("item_features"), obs_item.get("interactors")]) == key
        per_subject.setdefault(sid, []).append((float(label), same))
    res = []
    for obs in per_subject.values():
        rate = (sum(y for y, _ in obs) + 1.0) / (len(obs) + 2.0)
        res += [y - rate for y, same in obs if same]
    return SIB_GAMMA * float(np.mean(res)) if res else 0.0


def _predict(subject: dict, item: dict, labeled) -> float:
    start = time.monotonic()
    deadline = start + CALL_BUDGET
    bench = item.get("benchmark_id", "")
    lab_items, ys = [], []
    for (obs_subject, obs_item), label in labeled or []:
        if obs_subject == subject and obs_item.get("benchmark_id", "") == bench:
            lab_items.append(obs_item)
            ys.append(float(label))
    y = np.asarray(ys, dtype=float)
    items = [item] + lab_items
    keys, first = _unique(items)
    with ThreadPoolExecutor(max_workers=min(16, 2 * len(first))) as ex:
        jf = {k: ex.submit(_judge, subject, it, deadline) for k, it in first.items()}
        af = {k: ex.submit(_attempt, it, deadline) for k, it in first.items()} if len(y) else {}
        bf = {k: ex.submit(_attempt2, it, deadline) for k, it in first.items()} if len(y) else {}
        judged = [jf[k].result() for k in keys]
        tried = [af[k].result() for k in keys] if af else []
        tried2 = [bf[k].result() for k in keys] if bf else []
    target = judged[0]
    if target is None:
        return _fallback_gp(y)
    if len(y) == 0:
        x = np.array([target["z"], target["rel"], target["t"]])
        # Integrate benchmark-level uncertainty of the head (B0_TAU); 0 gives the plain logistic head.
        return float(_sigmoid(HEAD_C + HEAD_W @ x + B0_TAU * B0_X) @ B0_W)
    z_all = [j["z"] for j in judged if j is not None]
    z_fill = float(np.mean(z_all))
    z_l = np.array([j["z"] if j is not None else z_fill for j in judged[1:]])
    z_t = float(target["z"])

    def filled(vals):
        avail = [v for v in vals if v is not None]
        fill = float(np.mean(avail)) if avail else 0.0
        return [v if v is not None else fill for v in vals]
    def effort(a, b):
        vals = [-math.log1p(x["ot"]) for x in (a, b) if x is not None]
        return float(np.mean(vals)) if vals else None
    e_vals = filled([effort(a, b) for a, b in zip(tried, tried2 or [None] * len(tried))])
    a_vals = filled([None if (a is None or a.get("timeout") or "rt" not in a) else -math.log1p(max(0.0, a["ot"] - a["rt"]))
                     for a in tried])
    e_t, e_l = e_vals[0], np.array(e_vals[1:])
    a_t, a_l = a_vals[0], np.array(a_vals[1:])
    n = len(z_l)
    mu, zb_t = float(z_l.mean()), (z_t + float(z_l.sum())) / (n + 1)
    emu, eb_t = float(e_l.mean()), (e_t + float(e_l.sum())) / (n + 1)
    amu, ab_t = float(a_l.mean()), (a_t + float(a_l.sum())) / (n + 1)
    m_l = MUND_C + MUND_BB * mu + MUND_BW * (z_l - mu) + MUND_BE * (e_l - emu)
    m_t = MUND_C + MUND_BB * zb_t + MUND_BW * (z_t - zb_t) + MUND_BE * (e_t - eb_t)
    tok_l = [_tokens(it.get("item_features")) for it in lab_items]
    tok_t = _tokens(item.get("item_features"))
    extra = _sibling_shift(subject, item, bench, labeled)
    return _posterior_predict(m_l, z_l - mu, e_l - emu, (a_l - amu) / ANSWER_SCALE, y,
                              m_t, z_t - zb_t, e_t - eb_t, (a_t - ab_t) / ANSWER_SCALE, tok_l, tok_t, extra)


def _predict_lib(subject: dict, item: dict, labeled):
    """Public-results library (H22). None when the target item is not in the library."""
    if LIB is None:
        return None
    idx = LIB.match(item.get("item_content"))
    if idx is None:
        return None
    bench = item.get("benchmark_id", "")
    lab = [(obs_item, float(y)) for (obs_subject, obs_item), y in labeled or []
           if obs_subject == subject and obs_item.get("benchmark_id", "") == bench]
    lab_idx = [(LIB.match(it.get("item_content")), y) for it, y in lab]
    tgt = LIB.configs_for(idx, subject)
    lab_cfg = [LIB.configs_for(j, subject) if j is not None else {} for j, _ in lab_idx]
    keys = sorted(set(tgt).union(*[set(c) for c in lab_cfg]))
    p_easy = easy_predict(LIB.item_mean(idx), [(y, LIB.item_mean(j)) for j, y in lab_idx if j is not None])
    if any(k in tgt for k in keys):
        w, w0 = lib_priors([harness_match(f"{k[0]} | {k[1]}", subject) for k in keys], len(keys))
        rows = [(y, [c.get(k) for k in keys], LIB.item_mean(j) if j is not None else None) for (j, y), c in zip(lab_idx, lab_cfg)]
        p, _ = mix_predict([tgt.get(k) for k in keys], w, w0, p_easy, rows)   # fallback hypothesis = item easiness (H22 sim)
        return p
    return p_easy if EASY_MODE == "easy" else None


def predict(input: list, labeled: list | None = None) -> float:
    subject, item = input
    try:
        p = _predict_lib(subject, item, labeled)
    except Exception:
        p = None
    if p is None:
        p = _predict(subject, item, labeled)
    if not math.isfinite(p):
        p = float(_sigmoid(MU))
    return min(P_MAX, max(P_MIN, p))
