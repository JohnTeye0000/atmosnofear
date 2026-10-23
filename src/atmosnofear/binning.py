"""Aggregate raw trips into a regular-grid traffic series.

Sparse crowd-collected trips become one row per (segment, time bin) with the
median duration and the number of observations. Bins with no trips are simply
absent: missing is not zero (see docs/data_schema.md).
"""
from __future__ import annotations

import pandas as pd

SERIES_COLUMNS = ["segment_id", "bin_start_utc", "median_duration_s", "n_obs", "data_label"]


def bin_trips(trips: pd.DataFrame, bin_minutes: int = 15) -> pd.DataFrame:
    """Return one row per non-empty (segment, bin). Mixed data labels are refused."""
    if trips.empty:
        return pd.DataFrame(columns=SERIES_COLUMNS)
    labels = trips["data_label"].unique()
    if len(labels) > 1:
        raise ValueError(f"refusing to mix data labels in one series: {sorted(labels)}")
    start = pd.to_datetime(trips["start_time_utc"], utc=True)
    binned = trips.assign(bin_start_utc=start.dt.floor(f"{bin_minutes}min"))
    out = (binned.groupby(["segment_id", "bin_start_utc"])
           .agg(median_duration_s=("duration_s", "median"), n_obs=("duration_s", "size"))
           .reset_index())
    return out.assign(data_label=labels[0])[SERIES_COLUMNS]
