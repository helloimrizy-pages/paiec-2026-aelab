"""Load measurement-db into the long format used by sim.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent.parent / "data" / "measurement-db"
BENCHES = ("matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")
SUBJECT_FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness",
                  "reasoning_effort", "harness_version", "subject_features_extra"]


def load_long(benches=BENCHES, threshold: float = 0.5) -> pd.DataFrame:
    """One row per graded response, with y binarized at `threshold`.

    MMDocRAG items are split by test_condition, since the same question under
    a different retrieval setting is a different task for the subject.
    """
    parts = []
    for b in benches:
        r = pd.read_parquet(D / b / "response.parquet").dropna(subset=["response"])
        s = pd.read_parquet(D / b / "subjects.parquet")
        item = r["item_id"]
        if r["test_condition"].notna().any():
            item = item + "|" + r["test_condition"].fillna("")
        df = pd.DataFrame({"subject": r["subject_id"].to_numpy(), "benchmark": b, "item": item.to_numpy(),
                           "y": (r["response"].to_numpy() >= threshold).astype(float),
                           "y_raw": r["response"].to_numpy()})
        df = df.merge(s[["subject_id", "release_date", "normalized_name"]].rename(columns={"subject_id": "subject"}),
                      on="subject", how="left")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
