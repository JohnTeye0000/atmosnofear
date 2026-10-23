"""Simple, defensible baselines (research_plan.md section 8: B1 and B2)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MIN_CELL_COUNT = 3  # a (segment, weekday, time-of-day) cell needs this many training rows


def predict_persistence(examples: pd.DataFrame) -> pd.Series:
    """B2: assume the next travel time equals the most recent observed one."""
    return examples["last_obs_s"].astype(float)


@dataclass(frozen=True)
class HistoricalMean:
    """B1: mean training duration for the same segment, weekday and time of day.

    Falls back to the segment mean, then the overall mean, for sparse cells.
    Fitted on training rows only; the lookup tables are immutable after fitting.
    """
    cell_mean: dict
    segment_mean: dict
    overall_mean: float

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "HistoricalMean":
        keys = ["segment_id", "dow", "tod_min"]
        cells = train.groupby(keys)["target_duration_s"].agg(["mean", "size"])
        cells = cells[cells["size"] >= MIN_CELL_COUNT]
        return cls(cell_mean=cells["mean"].to_dict(),
                   segment_mean=train.groupby("segment_id")["target_duration_s"].mean().to_dict(),
                   overall_mean=float(train["target_duration_s"].mean()))

    def predict(self, examples: pd.DataFrame) -> pd.Series:
        def one(row) -> float:
            cell = (row.segment_id, row.dow, row.tod_min)
            if cell in self.cell_mean:
                return self.cell_mean[cell]
            return self.segment_mean.get(row.segment_id, self.overall_mean)
        values = [one(r) for r in examples.itertuples(index=False)]
        return pd.Series(values, index=examples.index, dtype=float)
