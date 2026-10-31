"""Simulator implementation of submission v3 with leave-one-benchmark-out calibration.

`V3` holds the pilot judgments and, for each held-out benchmark H, constants fitted on the
other pilot benchmarks. `V3.predictor(H)` returns a sim predictor for pairs of H;
`V3.prior_p(...)` exposes per-item quantities for acquisition policies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from data import load_long
from predictors import base_item
from tta import GPPair, KernelParams

TEST_LIKE = ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]
REL = {"much_easier": 2, "easier": 1, "typical": 0, "harder": -1, "much_harder": -2}


class V3:
    def __init__(self, tag: str = "luna_low", s0: float = 0.75, sz: float = 0.6):
        pp = pd.read_parquet("features/pilot_pairs.parquet")
        df = load_long(TEST_LIKE)
        df["item_id"] = df["item"].str.split("|").str[0]
        self.df = df.merge(pp, on=["subject", "benchmark", "item_id"])
        j = pd.read_parquet(f"features/apijudge_{tag}.parquet")
        self.J = {(r.subject, r.benchmark, r.item_id): (float(logit(np.clip(r.p_success, .01, .99))),
                                                        float(REL[r.relative_difficulty]),
                                                        float(logit(np.clip(r.typical_success_rate, .01, .99))))
                  for r in j.itertuples()}
        m = self.df.copy()
        m["z"] = [self.J[k][0] for k in zip(m.subject, m.benchmark, m.item_id)]
        m["rel"] = [self.J[k][1] for k in zip(m.subject, m.benchmark, m.item_id)]
        m["t"] = [self.J[k][2] for k in zip(m.subject, m.benchmark, m.item_id)]
        m["zbar"] = m.groupby(["subject", "benchmark"]).z.transform("mean")
        m["zdev"] = m.z - m.zbar
        self.m = m
        self.kp = KernelParams(s0, 0.0, 0.0, sz=sz)
        self.cal = {}
        for H in TEST_LIKE + [None]:
            tr = m if H is None else m[m.benchmark != H]
            w = 1.0 / tr.groupby(["subject", "benchmark"]).y.transform("size")
            head = LogisticRegression(C=100.0).fit(tr[["z", "rel", "t"]].to_numpy(), tr.y, sample_weight=w)
            mund = LogisticRegression(C=100.0).fit(tr[["zbar", "zdev"]].to_numpy(), tr.y, sample_weight=w)
            self.cal[H] = (head.intercept_[0], head.coef_[0], mund.intercept_[0], *mund.coef_[0])

    def z(self, s, b, item):
        return self.J[(s, b, base_item(item))][0]

    def predict_pair(self, H, s, b, target_items, lab_items, lab_y) -> np.ndarray:
        hc, hw, c, bB, bW = self.cal[H]
        jt = [self.J[(s, b, base_item(i))] for i in target_items]
        zt = np.array([x[0] for x in jt])
        if len(lab_items) == 0:
            return expit(hc + np.array(jt) @ hw)
        zl = np.array([self.z(s, b, i) for i in lab_items])
        zb_t = (zt + zl.sum()) / (len(zl) + 1)
        zb_l = zl.mean()
        empty = [frozenset()] * len(zl)
        gp = GPPair(empty, None, np.asarray(lab_y, float), c + bB * zb_l + bW * (zl - zb_l), self.kp, z=zl - zb_l)
        return gp.predict([frozenset()] * len(zt), None, c + bB * zb_t + bW * (zt - zb_t), z=zt - zb_t)

    def predictor(self, H):
        def f(t, lab):
            out = np.empty(len(t))
            groups = {k: g for k, g in lab.groupby(["subject", "benchmark"])} if len(lab) else {}
            for (s, b), g in t.groupby(["subject", "benchmark"]):
                lg = groups.get((s, b))
                li, ly = (lg["item"].tolist(), lg["y"].tolist()) if lg is not None else ([], [])
                out[g.index.to_numpy()] = self.predict_pair(H, s, b, g["item"].tolist(), li, ly)
            return out
        return f
