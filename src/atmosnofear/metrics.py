"""Error metrics and a block bootstrap for paired model comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def interval_coverage(y_true, lower, upper) -> float:
    """Share of true values inside [lower, upper]. For a q10-q90 interval the target is 0.80."""
    y = np.asarray(y_true, float)
    return float(np.mean((y >= np.asarray(lower, float)) & (y <= np.asarray(upper, float))))


def mean_interval_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))


def paired_mae_difference(y_true, pred_a, pred_b, blocks, n_boot: int = 2000, seed: int = 0,
                          alpha: float = 0.05) -> dict:
    """Mean of (|err_a| - |err_b|) with a block-bootstrap confidence interval.

    A negative mean means model A has the lower error. Consecutive observations are
    correlated, so whole blocks (e.g. days) are resampled instead of single rows.
    If the interval contains 0, the data do not show a difference.
    """
    diff = np.abs(np.asarray(y_true, float) - np.asarray(pred_a, float)) \
        - np.abs(np.asarray(y_true, float) - np.asarray(pred_b, float))
    grouped = pd.DataFrame({"d": diff, "b": np.asarray(blocks)}).groupby("b")["d"].agg(["sum", "size"])
    sums, counts = grouped["sum"].to_numpy(), grouped["size"].to_numpy()
    if len(sums) < 2:
        raise ValueError("need at least 2 blocks for a block bootstrap")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(len(diff)), "n_blocks": int(len(sums))}
