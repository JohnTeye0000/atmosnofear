"""Chronological train/validation/test split with a gap between folds."""
from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_split(examples: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2,
                        gap_min: int = 60) -> dict[str, pd.DataFrame]:
    """Split by prediction time t. Rows within `gap_min` before a boundary are dropped.

    The gap must be at least the largest horizon: a training row at t has its target
    at t + h, and that target must resolve before the next fold begins.
    """
    if not (0 < train_frac and 0 < val_frac and train_frac + val_frac < 1):
        raise ValueError("need 0 < train_frac, 0 < val_frac, train_frac + val_frac < 1")
    times = np.sort(examples["t_utc"].unique())
    if len(times) < 3:
        raise ValueError("need at least 3 distinct prediction times to split")
    cut_val = times[int(len(times) * train_frac)]
    cut_test = times[int(len(times) * (train_frac + val_frac))]
    gap = pd.Timedelta(minutes=gap_min).to_timedelta64()
    t = examples["t_utc"].to_numpy()
    masks = {
        "train": t < cut_val - gap,
        "val": (t >= cut_val) & (t < cut_test - gap),
        "test": t >= cut_test,
    }
    return {name: examples[m].reset_index(drop=True) for name, m in masks.items()}
