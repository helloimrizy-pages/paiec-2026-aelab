"""Reproducible LOBO experiments on answer-free cached inference signals.

Run from lab: .venv/bin/python ranking_model.py --variants all
Results are written to ../results/ranking/.
No network requests; attempt correctness and answers are never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, logsumexp
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression

import sim
from data import D
from predictors import base_item
from v3sim import TEST_LIKE, V3

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/ranking"
BUDG = [f"b{k}" for k in sim.BUDGETS]
VARIANTS = {
    "v3": dict(features=(), kind="v3"),
    "v4": dict(features=("effort",)),
    "v4_clip": dict(features=("effort",), clip=1.0),
    "answer_only": dict(features=("answer",)),
    "total_only": dict(features=("total",)),
    "effort_answer": dict(features=("effort", "answer")),
    "effort_fraction": dict(features=("effort", "fraction")),
    "effort_answer_clip": dict(features=("effort", "answer"), clip=1.0),
    "adaptive_effort": dict(features=("effort",), clip=1.0, adaptive=0.4),
    "adaptive_effort_answer": dict(features=("effort", "answer"), clip=1.0, adaptive=0.4),
    "effort_mixture": dict(features=("effort",), clip=1.0, reliability=True),
    "total_clip": dict(features=("total",), clip=1.0),
    "adaptive_total": dict(features=("total",), adaptive=0.4),
    "adaptive_total_clip": dict(features=("total",), clip=1.0, adaptive=0.4),
    "effort_total": dict(features=("effort", "total")),
    "total_quantile": dict(features=("totalq",)),
    "total_input": dict(features=("total", "input")),
    "average_effort_total": dict(features=(), kind="average"),
}


@dataclass
class PairEpisode:
    benchmark: str
    subject: str
    episode: int
    target_items: list[str]
    label_items: list[str]
    label_y: np.ndarray
    ymean: np.ndarray
    row_weights: np.ndarray


def posterior_grid(nodes: int = 41, slope_nodes: int = 21, adaptive=0.0, reliability=False, effort_nodes=9):
    """Gauss-Hermite prior, with exact mixture weights and normalization."""
    x, w = hermgauss(nodes)
    x, w = np.sqrt(2) * x, w / np.sqrt(np.pi)
    u = np.concatenate((0.5 * x, 2.0 * x))
    uw = np.concatenate((0.7 * w, 0.3 * w))
    sx, sw = hermgauss(slope_nodes)
    sx, sw = 0.6 * np.sqrt(2) * sx, sw / np.sqrt(np.pi)
    if adaptive:
        ex, ew = hermgauss(effort_nodes)
        ex, ew = adaptive * np.sqrt(2) * ex, ew / np.sqrt(np.pi)
    elif reliability:
        ex, ew = np.array([-1.0, 0.0, 1.0]), np.array([0.2, 0.6, 0.2])
    else:
        ex, ew = np.array([0.0]), np.array([1.0])
    uu, vv, ee = np.meshgrid(u, sx, ex, indexing="ij")
    pw = uw[:, None, None] * sw[None, :, None] * ew[None, None, :]
    return uu.ravel(), vv.ravel(), ee.ravel(), np.log(pw.ravel())


class RankingResearch:
    def __init__(self, nodes=41, slope_nodes=21, effort_nodes=9):
        self.v = V3()
        features = ["item_id", "sample", "reasoning_tokens", "output_tokens"]
        paths = [("attempt_med", "matharena"), ("attempt_ma_rest", None), ("attempt_other", None)]
        parts = []
        for name, benchmark in paths:
            cols = features + ([] if benchmark else ["benchmark"])
            d = pd.read_parquet(f"features/{name}.parquet", columns=cols)
            d = d[d["sample"] == 0].copy()
            if benchmark:
                d["benchmark"] = benchmark
            parts.append(d)
        a = pd.concat(parts).drop_duplicates(["benchmark", "item_id"])
        r = a.reasoning_tokens.to_numpy(float)
        o = a.output_tokens.to_numpy(float)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(o)) or np.any(o < r):
            raise ValueError("Invalid token usage in attempt cache")
        a["effort"] = -np.log1p(r)
        a["answer"] = -np.log1p(o - r)
        a["total"] = -np.log1p(o)
        a["fraction"] = r / np.maximum(o, 1.0)
        a["totalq"] = a["total"]  # runtime rank transform uses only visible items
        input_len = {}
        for b in TEST_LIKE:
            items = pd.read_parquet(D / b / "items.parquet", columns=["item_id", "content"])
            input_len.update({(b, row.item_id): np.log1p(len(row.content or "")) for row in items.itertuples()})
        a["input"] = [input_len[(b, i)] for b, i in zip(a.benchmark, a.item_id)]
        self.features = ["effort", "answer", "total", "fraction", "totalq", "input"]
        self.F = {(b, i): vals for b, i, *vals in a[["benchmark", "item_id"] + self.features].itertuples(index=False, name=None)}
        self.m = self.v.m.copy()
        for k, name in enumerate(self.features):
            self.m[name] = [self.F[(b, i)][k] for b, i in zip(self.m.benchmark, self.m.item_id)]
            self.m[name + "dev"] = self.m[name] - self.m.groupby(["subject", "benchmark"])[name].transform("mean")
        self.m["totalqdev"] = self.m.groupby(["subject", "benchmark"]).total.transform(
            lambda x: norm.ppf((rankdata(x, method="average") - 0.5) / len(x)))
        self.nodes, self.slope_nodes, self.effort_nodes = nodes, slope_nodes, effort_nodes
        self.cal = {}
        self.grids = {}

    def fit(self, H, variant):
        key = (H, variant)
        if key in self.cal:
            return self.cal[key]
        names = VARIANTS[variant]["features"]
        tr = self.m if H is None else self.m[self.m.benchmark != H]
        w = 1.0 / tr.groupby(["subject", "benchmark"]).y.transform("size")
        cols = ["zbar", "zdev"] + [x + "dev" for x in names]
        fit = LogisticRegression(C=100, max_iter=1000).fit(tr[cols].to_numpy(), tr.y, sample_weight=w)
        hc, hw = self.v.cal[H][:2]
        out = (hc, hw, float(fit.intercept_[0]), fit.coef_[0].copy())
        self.cal[key] = out
        return out

    def signals(self, benchmark, items, names):
        idx = [self.features.index(name) for name in names]
        return np.array([[self.F[(benchmark, base_item(i))][k] for k in idx] for i in items], dtype=float)

    def predict(self, H, variant, ep, budget):
        nt = len(ep.target_items)
        li = ep.label_items[:budget]
        y = ep.label_y[:budget]
        if variant == "v3":
            return self.v.predict_pair(H, ep.subject, ep.benchmark, ep.target_items, li, y)
        spec = VARIANTS[variant]
        if spec.get("kind") == "average":
            return 0.5 * (self.predict(H, "v4", ep, budget) + self.predict(H, "total_only", ep, budget))
        hc, hw, c, coefs = self.fit(H, variant)
        jt = np.array([self.v.J[(ep.subject, ep.benchmark, base_item(i))] for i in ep.target_items])
        if not budget:
            return expit(hc + jt @ hw)
        names = spec["features"]
        zt = jt[:, 0]
        zl = np.array([self.v.z(ep.subject, ep.benchmark, i) for i in li])
        ft, fl = self.signals(ep.benchmark, ep.target_items, names), self.signals(ep.benchmark, li, names)
        n = len(zl)
        zmu, zb_t = zl.mean(), (zt + zl.sum()) / (n + 1)
        fmu, fb_t = fl.mean(0), (ft + fl.sum(0)) / (n + 1)
        dl, dt = fl - fmu, ft - fb_t
        for k, name in enumerate(names):
            if name == "totalq":
                dl[:, k] = norm.ppf((rankdata(fl[:, k], method="average") - 0.5) / n)
                ranks = (fl[:, k, None] < ft[:, k]).sum(0) + 0.5 * ((fl[:, k, None] == ft[:, k]).sum(0) + 1)
                dt[:, k] = norm.ppf(ranks / (n + 1))
        if spec.get("clip"):
            dl, dt = np.clip(dl, -spec["clip"], spec["clip"]), np.clip(dt, -spec["clip"], spec["clip"])
        bB, bW, *bF = coefs
        bF = np.array(bF)
        ml = c + bB * zmu + bW * (zl - zmu) + dl @ bF
        mt = c + bB * zb_t + bW * (zt - zb_t) + dt @ bF
        grid_key = (spec.get("adaptive", 0.0), spec.get("reliability", False))
        if grid_key not in self.grids:
            self.grids[grid_key] = posterior_grid(self.nodes, self.slope_nodes, *grid_key, effort_nodes=self.effort_nodes)
        U, V, E, LW = self.grids[grid_key]
        edl, edt = dl[:, 0], dt[:, 0]
        if spec.get("reliability"):
            edl, edt = bF[0] * edl, bF[0] * edt
        ll = ml[:, None] + U[None, :] + (zl - zmu)[:, None] * V[None, :] + edl[:, None] * E[None, :]
        lw = LW + (y[:, None] * ll - np.logaddexp(0, ll)).sum(0)
        post = np.exp(lw - logsumexp(lw))
        # Chunking bounds memory for adaptive grids.
        out = np.empty(nt)
        for st in range(0, nt, 24):
            sl = slice(st, st + 24)
            out[sl] = expit(mt[sl, None] + U[None, :] + (zt[sl] - zb_t[sl])[:, None] * V[None, :] + edt[sl, None] * E[None, :]) @ post
        return out

    def episodes(self, splits, seed):
        episodes = []
        for H in TEST_LIKE:
            df = self.v.df[self.v.df.benchmark == H]
            pairs = sim.eligible_pairs(df)
            rng = np.random.default_rng(seed)
            for k in range(splits):
                e = sim.make_episode(df, pairs, rng)
                for (s, b), target in e.eval_rows.groupby(["subject", "benchmark"], sort=True):
                    # Preserve source order used by the simulator.
                    ids = target["item"].drop_duplicates().tolist()
                    grouped = target.groupby("item").y.agg(["mean", "size"]).loc[ids]
                    lab = e.acq_stream[(e.acq_stream.subject == s) & (e.acq_stream.benchmark == b)].sort_values("rank")
                    episodes.append(PairEpisode(b, s, k, ids, lab["item"].tolist(), lab.y.to_numpy(float),
                                                grouped["mean"].to_numpy(float), grouped["size"].to_numpy(float) / grouped["size"].sum()))
        return episodes


def boot_delta(a, b, n=4000):
    d = a.merge(b, on=["benchmark", "subject"], suffixes=("", "_control"))
    groups = [(g.alc - g.alc_control).to_numpy() for _, g in d.groupby("benchmark")]
    rng = np.random.default_rng(0)
    boots = np.mean([rng.choice(x, (n, len(x))).mean(1) for x in groups], axis=0)
    return dict(delta=float(np.mean([x.mean() for x in groups])), ci95=np.percentile(boots, [2.5, 97.5]).tolist(),
                pairs_improved=int(sum((x < 0).sum() for x in groups)), pairs=int(sum(len(x) for x in groups)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="all")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--nodes", type=int, default=41)
    ap.add_argument("--slope-nodes", type=int, default=21)
    ap.add_argument("--effort-nodes", type=int, default=9)
    ap.add_argument("--tag", default="primary")
    args = ap.parse_args()
    variants = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    for name in variants:
        if name not in VARIANTS:
            ap.error(f"Unknown variant: {name}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    research = RankingResearch(args.nodes, args.slope_nodes, args.effort_nodes)
    eps = research.episodes(args.splits, args.seed)
    records, summaries, pair_results = [], {}, {}
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in variants:
        started = time.monotonic()
        rows = []
        for k, ep in enumerate(eps):
            scores = []
            for budget in sim.BUDGETS:
                p = research.predict(ep.benchmark, name, ep, budget)
                if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
                    raise ValueError("Invalid predictions")
                # Includes Bernoulli variance for repeated responses; do not square a mean label.
                scores.append(float((p * p - 2 * p * ep.ymean + ep.ymean) @ ep.row_weights))
            rows.append(dict(variant=name, benchmark=ep.benchmark, subject=ep.subject, episode=ep.episode,
                             **dict(zip(BUDG, scores)), alc=float(np.array(scores) @ sim.WEIGHTS)))
            if (k + 1) % 120 == 0:
                print(f"{name}: {k + 1}/{len(eps)} pair-splits ({time.monotonic() - started:.1f}s)", flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(RESULTS / f"{args.tag}_{name}_episodes.csv", index=False)
        pp = frame.groupby(["benchmark", "subject"])[BUDG + ["alc"]].mean().reset_index()
        pp.to_csv(RESULTS / f"{args.tag}_{name}_pairs.csv", index=False)
        pair_results[name] = pp
        cv = pp.groupby("benchmark")[BUDG + ["alc"]].mean()
        summ = dict(variant=name, alc=float(cv.alc.mean()), curve=cv.mean().to_dict(),
                    per_benchmark=cv.alc.to_dict(), seconds=time.monotonic() - started,
                    source_sha256=source_sha, seed=args.seed, splits=args.splits,
                    integration_nodes=[args.nodes, args.slope_nodes, args.effort_nodes],
                    coefs={str(H): (research.fit(H, name)[2], research.fit(H, name)[3].tolist())
                           for H in TEST_LIKE} if not VARIANTS[name].get("kind") else {})
        for control in ["v3", "v4", "v4_clip"]:
            if control in pair_results and control != name:
                summ["vs_" + control] = boot_delta(pp, pair_results[control])
        summaries[name] = summ
        (RESULTS / f"{args.tag}_summary.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps(summ), flush=True)
        records.extend(rows)
    pd.DataFrame(records).to_csv(RESULTS / f"{args.tag}_trajectory.csv", index=False)


if __name__ == "__main__":
    main()
