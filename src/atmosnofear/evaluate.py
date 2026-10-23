"""Run the baselines across horizons and report errors per split."""
from __future__ import annotations

import pandas as pd

from .baselines import HistoricalMean, predict_persistence
from .features import build_examples
from .metrics import mae, rmse
from .split import chronological_split

DEFAULT_HORIZONS = (15, 30, 60)


def evaluate_baselines(series: pd.DataFrame, horizons=DEFAULT_HORIZONS, bin_minutes: int = 15) -> pd.DataFrame:
    """MAE/RMSE (in minutes) for each baseline, horizon and split.

    The split gap is the largest horizon so no training target overlaps the next fold.
    Every row carries the series data_label so synthetic results cannot pass as real.
    """
    labels = series["data_label"].unique()
    if len(labels) != 1:
        raise ValueError(f"series must have exactly one data_label, got {sorted(labels)}")
    gap = max(horizons)
    rows = []
    for h in horizons:
        folds = chronological_split(build_examples(series, h, bin_minutes), gap_min=gap)
        model = HistoricalMean.fit(folds["train"])
        for split in ("val", "test"):
            ex = folds[split]
            if ex.empty:
                continue
            for name, pred in (("historical_mean", model.predict(ex)), ("persistence", predict_persistence(ex))):
                rows.append({"horizon_min": h, "split": split, "model": name, "n": len(ex),
                             "mae_min": mae(ex["target_duration_s"], pred) / 60,
                             "rmse_min": rmse(ex["target_duration_s"], pred) / 60,
                             "data_label": labels[0]})
    return pd.DataFrame(rows)
