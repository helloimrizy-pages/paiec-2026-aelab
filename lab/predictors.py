"""Predictors for the simulator, built from per-item lookups and the TTA GP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from data import BENCHES, D
from tta import GPPair, KernelParams, feature_tokens

HERE = Path(__file__).resolve().parent


def base_item(key: str) -> str:
    return key.split("|")[0]


class ItemStore:
    """Item tokens and normalized embeddings keyed by (benchmark, item_id)."""

    def __init__(self, emb_tag: str | None = "qwen3e06"):
        it = pd.concat([pd.read_parquet(D / b / "items.parquet", columns=["item_id", "benchmark_id", "item_features"])
                        for b in BENCHES], ignore_index=True)
        self.tok = {(b, i): feature_tokens(f) for b, i, f in zip(it.benchmark_id, it.item_id, it.item_features.fillna(""))}
        self.emb = None
        if emb_tag:
            e = pd.read_parquet(HERE / "features" / f"emb_{emb_tag}.parquet")
            self.emb = {(b, i): np.asarray(v, dtype=np.float64) for b, i, v in zip(e.benchmark, e.item_id, e.emb)}

    def arrays(self, bench: str, items):
        keys = [(bench, base_item(i)) for i in items]
        tok = [self.tok.get(k, frozenset()) for k in keys]
        emb = np.stack([self.emb[k] for k in keys]) if self.emb is not None else None
        return tok, emb


def gp_predictor(store: ItemStore, kp: KernelParams, prior_p: float = 0.4, prior_mean_fn=None, z_fn=None):
    """Per-pair GP using only that pair's labels.

    prior_mean_fn(frame) -> logit prior mean per row (optional); default is a
    constant logit(prior_p).
    """
    def f(t: pd.DataFrame, lab: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(t))
        lab_groups = {k: g for k, g in lab.groupby(["subject", "benchmark"])} if len(lab) else {}
        for (s, b), g in t.groupby(["subject", "benchmark"]):
            tok_t, emb_t = store.arrays(b, g["item"])
            m_t = prior_mean_fn(g) if prior_mean_fn else np.full(len(g), logit(prior_p))
            z_t = z_fn(g) if z_fn else None
            lg = lab_groups.get((s, b))
            if lg is None or len(lg) == 0:
                gp = GPPair([], None, np.zeros(0), np.zeros(0), kp)
            else:
                tok_l, emb_l = store.arrays(b, lg["item"])
                m_l = prior_mean_fn(lg) if prior_mean_fn else np.full(len(lg), logit(prior_p))
                gp = GPPair(tok_l, emb_l, lg["y"].to_numpy(), m_l, kp, z=z_fn(lg) if z_fn else None)
            out[g.index.to_numpy()] = gp.predict(tok_t, emb_t, m_t, z_t)
        return out
    return f
