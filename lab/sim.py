"""Local replica of the PAIEC streaming evaluation protocol.

Follows the rules in the starting kit README:
- A (subject, benchmark) pair is eligible when it has >= 80 distinct items.
- A draw picks complete pairs under a cap of 1,000 distinct subject-item
  pairs, preferring benchmarks and subjects drawn less often, random ties.
  Pairs that do not fit in the remaining cap are skipped.
- Each pair's distinct items are split 50/50 into acquisition and evaluation
  pools (odd counts give the extra item to acquisition).
- Default random acquisition reveals one recorded response per chosen item;
  budget-k evidence is the first k acquired items of each pair (nested).
- `labeled` at a budget holds the evidence of every drawn pair, not only the
  target pair.
- Brier is computed over every response row of the evaluation items; a pair's
  ALC weights budgets 0,1,3,7,15,31 by .1,.2,.2,.2,.2,.1; pairs are averaged
  with equal weight.

Data frame contract: columns subject, benchmark, item, y (0/1 float), one row
per recorded response. Extra columns are passed through to predictors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
MIN_ITEMS = 80
DRAW_CAP = 1000


def eligible_pairs(df: pd.DataFrame, min_items: int = MIN_ITEMS) -> pd.DataFrame:
    n = df.groupby(["subject", "benchmark"])["item"].nunique().rename("n_items")
    return n[n >= min_items].reset_index()


def draw_pairs(pairs: pd.DataFrame, rng: np.random.Generator, cap: int = DRAW_CAP) -> pd.DataFrame:
    """Greedy draw that prefers less-represented benchmarks and subjects."""
    remaining = cap
    left = pairs.sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)
    chosen, bcount, scount = [], {}, {}
    while len(left):
        key = left["benchmark"].map(lambda b: bcount.get(b, 0)) + left["subject"].map(lambda s: scount.get(s, 0))
        fits = left["n_items"] <= remaining
        if not fits.any():
            break
        cand = left[fits & (key == key[fits].min())]
        row = cand.iloc[int(rng.integers(len(cand)))]
        chosen.append(row)
        remaining -= int(row["n_items"])
        bcount[row["benchmark"]] = bcount.get(row["benchmark"], 0) + 1
        scount[row["subject"]] = scount.get(row["subject"], 0) + 1
        left = left.drop(index=row.name)
    return pd.DataFrame(chosen).reset_index(drop=True)


@dataclass
class Episode:
    """One drawn set of pairs with fixed splits and acquisition order."""

    eval_rows: pd.DataFrame  # every response row of evaluation items
    acq_stream: pd.DataFrame  # one revealed response per acquired item, ordered by `rank` within pair


def make_episode(df: pd.DataFrame, pairs: pd.DataFrame, rng: np.random.Generator,
                 max_labels: int = max(BUDGETS)) -> Episode:
    sub = df.merge(pairs[["subject", "benchmark"]], on=["subject", "benchmark"])
    eval_parts, acq_parts = [], []
    for (s, b), g in sub.groupby(["subject", "benchmark"], sort=False):
        items = g["item"].unique()
        items = items[rng.permutation(len(items))]
        n_acq = (len(items) + 1) // 2
        acq_items, eval_items = items[:n_acq], set(items[n_acq:])
        eval_parts.append(g[g["item"].isin(eval_items)])
        # Random acquisition: the first max_labels items of a random stream order.
        picked = acq_items[:max_labels]
        rows = g[g["item"].isin(picked)]
        # Reveal one recorded response per acquired item.
        rows = rows.sample(frac=1.0, random_state=int(rng.integers(2**31))).drop_duplicates("item")
        order = {it: r for r, it in enumerate(picked)}
        rows = rows.assign(rank=rows["item"].map(order)).sort_values("rank")
        acq_parts.append(rows)
    return Episode(pd.concat(eval_parts, ignore_index=True), pd.concat(acq_parts, ignore_index=True))


def score_episode(ep: Episode, predictor, budgets=BUDGETS) -> pd.DataFrame:
    """Return per-pair Brier at each budget plus ALC.

    `predictor(targets, labeled)` gets the distinct evaluation (subject, item)
    rows and the evidence frame, and returns one probability per target row.
    """
    targets = ep.eval_rows.drop_duplicates(["subject", "benchmark", "item"]).drop(columns=["y"]).reset_index(drop=True)
    out = []
    for k in budgets:
        labeled = ep.acq_stream[ep.acq_stream["rank"] < k]
        p = np.asarray(predictor(targets, labeled), dtype=float)
        if p.shape != (len(targets),) or not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
            raise ValueError("predictor returned invalid probabilities")
        pred = targets[["subject", "benchmark", "item"]].assign(p=p)
        rows = ep.eval_rows.merge(pred, on=["subject", "benchmark", "item"])
        rows["se"] = (rows["p"] - rows["y"]) ** 2
        b = rows.groupby(["subject", "benchmark"])["se"].mean().rename(f"b{k}")
        out.append(b)
    res = pd.concat(out, axis=1).reset_index()
    res["alc"] = res[[f"b{k}" for k in budgets]].to_numpy() @ WEIGHTS
    return res


def evaluate(df: pd.DataFrame, predictor, n_episodes: int = 20, seed: int = 0,
             use_draw: bool = True, pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Average per-pair results over several random episodes."""
    rng = np.random.default_rng(seed)
    pairs = eligible_pairs(df) if pairs is None else pairs
    res = []
    for e in range(n_episodes):
        drawn = draw_pairs(pairs, rng) if use_draw else pairs
        ep = make_episode(df, drawn, rng)
        r = score_episode(ep, predictor)
        r["episode"] = e
        res.append(r)
    return pd.concat(res, ignore_index=True)


def summarize(res: pd.DataFrame) -> pd.Series:
    """Mean over pairs within an episode, then mean and s.e. over episodes."""
    cols = [f"b{k}" for k in BUDGETS if f"b{k}" in res.columns] + ["alc"]
    per_ep = res.groupby("episode")[cols].mean()
    out = per_ep.mean()
    out["alc_se"] = per_ep["alc"].std(ddof=1) / np.sqrt(len(per_ep)) if len(per_ep) > 1 else np.nan
    return out


def make_stream_episode(df: pd.DataFrame, pairs: pd.DataFrame, rng: np.random.Generator):
    """Fixed splits plus a random candidate order per pair; labels are chosen later by a policy."""
    sub = df.merge(pairs[["subject", "benchmark"]], on=["subject", "benchmark"])
    eval_parts, streams = [], {}
    for (s, b), g in sub.groupby(["subject", "benchmark"], sort=False):
        items = g["item"].unique()
        items = items[rng.permutation(len(items))]
        n_acq = (len(items) + 1) // 2
        acq_items, eval_items = items[:n_acq], set(items[n_acq:])
        eval_parts.append(g[g["item"].isin(eval_items)])
        # One recorded response per candidate, revealed only if acquired.
        rows = g[g["item"].isin(acq_items)].sample(frac=1.0, random_state=int(rng.integers(2**31))).drop_duplicates("item")
        order = {it: r for r, it in enumerate(acq_items)}
        streams[(s, b)] = rows.assign(pos=rows["item"].map(order)).sort_values("pos").reset_index(drop=True)
    return pd.concat(eval_parts, ignore_index=True), streams


def score_streaming(eval_rows: pd.DataFrame, streams: dict, predictor, policy, budgets=BUDGETS,
                    max_labels: int = max(BUDGETS)) -> pd.DataFrame:
    """Streaming acquisition: `policy(cand_row, acquired_frame, context) -> bool`.

    All pairs reach a checkpoint before any pair acquires more. If the stream
    runs out, later checkpoints use the labels acquired so far.
    """
    targets = eval_rows.drop_duplicates(["subject", "benchmark", "item"]).drop(columns=["y"]).reset_index(drop=True)
    pos = {k: 0 for k in streams}
    acquired = {k: [] for k in streams}
    out = []
    for k in budgets:
        for key, st in streams.items():
            while len(acquired[key]) < min(k, max_labels) and pos[key] < len(st):
                cand = st.iloc[pos[key]]
                ctx = {"labels_remaining": max_labels - len(acquired[key]), "labels_acquired": len(acquired[key]),
                       "max_labels": max_labels, "items_remaining": len(st) - pos[key]}
                acq = st.iloc[acquired[key]] if acquired[key] else st.iloc[:0]
                take = ctx["items_remaining"] <= ctx["labels_remaining"] or bool(policy(cand, acq, ctx))
                if take:
                    acquired[key].append(pos[key])
                pos[key] += 1
        labeled = pd.concat([st.iloc[acquired[key]] for key, st in streams.items()], ignore_index=True)
        p = np.asarray(predictor(targets, labeled), dtype=float)
        pred = targets[["subject", "benchmark", "item"]].assign(p=p)
        rows = eval_rows.merge(pred, on=["subject", "benchmark", "item"])
        rows["se"] = (rows["p"] - rows["y"]) ** 2
        out.append(rows.groupby(["subject", "benchmark"])["se"].mean().rename(f"b{k}"))
    res = pd.concat(out, axis=1).reset_index()
    res["alc"] = res[[f"b{k}" for k in budgets]].to_numpy() @ WEIGHTS
    return res


def random_policy(rng: np.random.Generator):
    """Platform default: accept with probability labels_remaining / items_remaining."""
    return lambda cand, acq, ctx: rng.random() < ctx["labels_remaining"] / ctx["items_remaining"]
