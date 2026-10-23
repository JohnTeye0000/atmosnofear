"""The main experiment: do weather features reduce error, all else equal?

For each horizon the SAME train/val/test split is used for every model and feature set:
  traffic            history + calendar only
  traffic+forecast   adds archived-forecast weather (deployable; label nwp_forecast)
  traffic+oracle     adds reanalysis weather (NOT deployable; an upper bound; label reanalysis)
The headline comparison is traffic vs traffic+forecast for the same model type, with a
paired block bootstrap on the absolute-error difference (blocks = days).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import HistoricalMean, predict_persistence
from .features import add_weather, build_examples
from .metrics import mae, paired_mae_difference, rmse
from .models import TRAFFIC_COLS, WEATHER_COLS, fit_gbm, fit_mlp
from .split import chronological_split

LOW_RAIN_WARNING_BINS = 100   # fewer rainy test examples than this -> little power to see an effect
MIN_TEST_DAYS = 20            # fewer test days than this -> too few bootstrap blocks to trust the interval


def _verdict(ci_low: float, ci_high: float) -> str:
    if ci_high < 0:
        return "weather lowers error"
    if ci_low > 0:
        return "weather raises error"
    return "no difference detected"


def run_weather_experiment(series: pd.DataFrame, weather_fc: pd.DataFrame | None = None,
                           weather_oracle: pd.DataFrame | None = None, horizons=(15, 30, 60),
                           bin_minutes: int = 15, seeds=(0, 1, 2), mlp_epochs: int = 200,
                           weather_provider: pd.DataFrame | None = None, tz: str = "UTC",
                           mlp_batch_size: int = 256):
    """Return (results, comparisons, notes). results/comparisons carry the data label.

    weather_provider is a dataset's own observed-style weather (label provider_weather):
    like the oracle set, it is not a forecast, so it is an upper bound and not deployable.
    """
    labels = series["data_label"].unique()
    if len(labels) != 1:
        raise ValueError(f"series must have exactly one data_label, got {sorted(labels)}")
    label = labels[0]
    feature_sets = {"traffic": list(TRAFFIC_COLS)}
    if weather_fc is not None:
        feature_sets["traffic+forecast"] = TRAFFIC_COLS + WEATHER_COLS["forecast"]
    if weather_oracle is not None:
        feature_sets["traffic+oracle"] = TRAFFIC_COLS + WEATHER_COLS["oracle"]
    if weather_provider is not None:
        feature_sets["traffic+provider"] = TRAFFIC_COLS + WEATHER_COLS["provider"]

    results, comparisons, notes = [], [], []
    for h in horizons:
        ex = build_examples(series, h, bin_minutes, tz=tz)
        for mode, w in (("forecast", weather_fc), ("oracle", weather_oracle), ("provider", weather_provider)):
            if w is not None:
                ex = add_weather(ex, w, mode)
        folds = chronological_split(ex, gap_min=max(horizons))
        train, val, test = folds["train"], folds["val"], folds["test"]
        y = test["target_duration_s"].to_numpy()
        blocks = test["target_bin_start"].dt.date.to_numpy()

        preds = {("historical_mean", "calendar"): HistoricalMean.fit(train).predict(test).to_numpy(),
                 ("persistence", "last_obs"): predict_persistence(test).to_numpy()}
        for name, cols in feature_sets.items():
            preds[("gbm", name)] = fit_gbm(train, val, cols).predict(test)
            nets = [fit_mlp(train, val, cols, seed=s, max_epochs=mlp_epochs, batch_size=mlp_batch_size).predict(test)
                    for s in seeds]
            preds[("mlp", name)] = np.mean(nets, axis=0)   # ensemble over seeds

        n_days = len(set(blocks))
        if n_days < MIN_TEST_DAYS:
            notes.append(f"horizon {h}: only {n_days} test days; block-bootstrap intervals are unreliable below {MIN_TEST_DAYS}")
        rainy = int(test["wx_fc_rain"].sum()) if "wx_fc_rain" in test else None
        if rainy is not None and rainy < LOW_RAIN_WARNING_BINS:
            notes.append(f"horizon {h}: only {rainy} rainy test examples; the weather comparison has little power")
        for (model, fs), p in preds.items():
            results.append({"horizon_min": h, "model": model, "feature_set": fs, "n_test": len(test),
                            "test_rainy": rainy, "mae_s": mae(y, p), "rmse_s": rmse(y, p),
                            "mae_min": mae(y, p) / 60, "rmse_min": rmse(y, p) / 60, "data_label": label})
        for model in ("gbm", "mlp"):
            for fs in feature_sets:
                if fs == "traffic":
                    continue
                try:
                    r = paired_mae_difference(y, preds[(model, fs)], preds[(model, "traffic")], blocks)
                except ValueError as err:
                    comparisons.append({"horizon_min": h, "model": model, "with": fs, "verdict": str(err),
                                        "data_label": label})
                    continue
                comparisons.append({"horizon_min": h, "model": model, "with": fs,
                                    "mean_diff_s": r["mean_diff"], "ci_low_s": r["ci_low"], "ci_high_s": r["ci_high"],
                                    "mean_diff_min": r["mean_diff"] / 60, "ci_low_min": r["ci_low"] / 60,
                                    "ci_high_min": r["ci_high"] / 60, "n": r["n"], "n_days": r["n_blocks"],
                                    "verdict": _verdict(r["ci_low"], r["ci_high"]), "data_label": label})
    if comparisons:
        notes.append(f"{len(comparisons)} comparisons were run at 95% intervals, so about "
                     f"{max(1, round(0.05 * len(comparisons)))} 'significant' result(s) are expected by chance alone; "
                     "treat any single flagged cell with caution")
    return pd.DataFrame(results), pd.DataFrame(comparisons), notes
