"""Runtime public-results library (H22). numpy + json only. Tokenizer, shingles and hashes are identical to
lab/h22_library.py (checked by eval/check_v9_equivalence.py).

A hidden item is matched to a library item by voting over hashed 8-token shingles of every text value of its
item_content (JSON wrappers are parsed). Library items carry per-(model, config) recorded outcomes from public,
inventory-listed releases; model names are measurement-db canonical names (same registry as `normalized_name`).
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import numpy as np

W, SHORT_MIN = 8, 5
TOK = re.compile(r"[a-z0-9]+(?:\+\+|#)?")   # keeps c++ / c# / f# distinct


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def texts(content) -> list:
    s = "" if content is None else str(content)
    out = [s]
    try:
        stack = [json.loads(s)]
        while stack:
            o = stack.pop()
            if isinstance(o, str): out.append(o)
            elif isinstance(o, dict): stack.extend(o.values())
            elif isinstance(o, list): stack.extend(o)
    except Exception:
        pass
    return out


def shingles(tokens: list) -> list:
    if len(tokens) < W:
        return [" ".join(tokens)] if len(tokens) >= SHORT_MIN else []
    return [" ".join(tokens[i:i + W]) for i in range(len(tokens) - W + 1)]


def query_shingles(content) -> np.ndarray:
    hs = set()
    for t in texts(content):
        tk = TOK.findall(t.lower())
        hs.update(h64(x) for x in shingles(tk))
        for w in range(SHORT_MIN, W):
            hs.update(h64(" ".join(tk[i:i + w])) for i in range(len(tk) - w + 1))
    return np.fromiter(hs, dtype=np.int64)


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


class PubLib:
    def __init__(self, path):
        p = Path(path)
        self.h = np.load(p / "sh_hash.npy"); self.i = np.load(p / "sh_item.npy")
        self.items = json.load(open(p / "items.json"))                  # [slug, n_sh, mean, n]
        self.outs = json.load(open(p / "outcomes.json"))               # idx -> [[canon, model, config, mean, size], ...]
        self._m = {}

    def match(self, content, min_votes: int = 3, min_frac: float = 0.5):
        key = hashlib.sha1(str(content).encode("utf-8", "ignore")).hexdigest()
        if key in self._m: return self._m[key]
        q = query_shingles(content); res = None
        if len(q):
            lo = np.searchsorted(self.h, q, "left"); hi = np.searchsorted(self.h, q, "right")
            parts = [self.i[a:b] for a, b in zip(lo, hi) if b > a]
            if parts:
                ids, votes = np.unique(np.concatenate(parts), return_counts=True)
                j = int(np.argmax(votes)); idx, v = int(ids[j]), int(votes[j]); n_sh = self.items[idx][1]
                if int((votes == v).sum()) == 1 and v >= min(min_votes, n_sh) and v / n_sh >= min_frac: res = idx
        self._m[key] = res
        return res

    def item_mean(self, idx):
        _, _, mean, n = self.items[idx]
        return (mean * n + 0.5) / (n + 1.0)                            # shrink toward 0.5 with one pseudo-observation

    def configs_for(self, idx, subject: dict) -> dict:
        """Recorded outcomes on item idx of the subject's canonical model, keyed by (model, config)."""
        name = norm(subject.get("normalized_name"))
        if not name: return {}
        return {(m, c): mean for canon, m, c, mean, size in self.outs.get(str(idx), []) if canon and norm(canon) == name}


EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")


def _effort_of(text: str):
    """Reasoning-effort word in a library label, e.g. 'GPT-OSS-20B (High)' -> 'high'; longest word wins (xhigh over high)."""
    words = re.findall(r"[a-z]+", str(text or "").lower())
    found = [e for e in EFFORTS if e in words]
    return max(found, key=len) if found else None


def harness_match(label: str, subject: dict) -> bool:
    """A library run (model string + config label) matches the subject's run settings when the subject's harness (and
    harness version, if given) appear in it and any reasoning effort it states agrees with the subject's; with no harness
    on the subject, an explicitly matching reasoning effort counts as a match."""
    c = norm(label); h = norm(subject.get("harness")); e = norm(subject.get("reasoning_effort")); le = _effort_of(label)
    if h:
        if h not in c: return False
        v = norm(subject.get("harness_version"))
        if v and v not in c: return False
        return not (e and le and le != e)
    return bool(e and le and le == e)
