"""Model zoo experiment: many architectures, same split, same tuning, with vs without weather.

Reference models (no history beyond the last value):   gbm, mlp           (level target)
New models (add `n_lags` of travel-time history):      gbm_resid, mlp_resid, gru_resid (residual target)
Combination:                                           blend = mean(gbm_resid, mlp_resid, gru_resid)
Uncertainty:                                           gbm_q, mlp_q (q10 / q50 / q90 intervals)

Selection discipline: every model is scored on the VALIDATION fold too. The model called
"selected" per horizon is the traffic-only model with the lowest validation MAE. Test results
are reported for all models and are never used to choose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import HistoricalMean, predict_persistence
from .deep import fit_gru, fit_quantile_mlp
from .experiment import _verdict
from .features import add_weather, build_examples, lag_columns
from .metrics import interval_coverage, mae, mean_interval_width, paired_mae_difference, rmse
from .models import TRAFFIC_COLS, WEATHER_COLS, fit_gbm, fit_gbm_quantiles, fit_mlp
from .split import chronological_split

MODELS = ("gbm", "mlp", "gbm_resid", "mlp_resid", "gru_resid", "blend", "gbm_q", "mlp_q")


def shuffle_days(weather: pd.DataFrame, seed: int = 0, tz: str = "UTC") -> pd.DataFrame:
    """PLACEBO weather: each local day receives the weather of a different, randomly chosen day.

    Marginal distribution and daily/hourly shape are preserved, but any true link between the
    weather and that day's traffic is destroyed. If the real forecast does not beat this placebo,
    a measured 'gain' from weather features is not evidence of a weather effect.
    """
    local = weather["valid_time_utc"].dt.tz_convert(tz)
    key = pd.DataFrame({"day": local.dt.date.to_numpy(), "hour": local.dt.hour.to_numpy()})
    days = np.array(sorted(key["day"].unique()))
    source_day = dict(zip(days, np.random.default_rng(seed).permutation(days)))
    cols = [c for c in ("precipitation", "temperature_2m") if c in weather.columns]
    lookup = weather[cols].assign(day=key["day"].to_numpy(), hour=key["hour"].to_numpy()).set_index(["day", "hour"])
    lookup = lookup[~lookup.index.duplicated()]
    src = pd.MultiIndex.from_arrays([key["day"].map(source_day), key["hour"]])
    shuffled = lookup.reindex(src)
    return pd.DataFrame({"valid_time_utc": weather["valid_time_utc"].to_numpy(),
                         **{c: shuffled[c].to_numpy() for c in cols}, "data_label": "placebo_weather"})


def _fit_models(train, val, test, cols, lag_cols, seeds, epochs, batch):
    """Fit every model on one feature set. Returns {name: (val_pred, test_pred)} and interval arrays."""
    with_lags = cols + lag_cols
    out, intervals = {}, {}

    def both(p):
        return p.predict(val), p.predict(test)

    gbm = fit_gbm(train, val, cols)
    out["gbm"] = both(gbm)
    gbm_r = fit_gbm(train, val, with_lags, residual=True)
    out["gbm_resid"] = both(gbm_r)

    def seeded(fit):
        preds = [both(fit(s)) for s in seeds]
        return np.mean([p[0] for p in preds], axis=0), np.mean([p[1] for p in preds], axis=0)

    out["mlp"] = seeded(lambda s: fit_mlp(train, val, cols, seed=s, max_epochs=epochs, batch_size=batch, patience=5))
    out["mlp_resid"] = seeded(lambda s: fit_mlp(train, val, with_lags, seed=s, max_epochs=epochs, batch_size=batch,
                                                patience=5, residual=True))
    out["gru_resid"] = seeded(lambda s: fit_gru(train, val, cols, lag_cols, seed=s, max_epochs=epochs,
                                                batch_size=batch))
    out["blend"] = tuple(np.mean([out[m][i] for m in ("gbm_resid", "mlp_resid", "gru_resid")], axis=0)
                         for i in (0, 1))

    gq = fit_gbm_quantiles(train, with_lags, gbm_r.params)
    out["gbm_q"] = both(gq)
    intervals["gbm_q"] = gq.predict_quantiles(test)
    mq = fit_quantile_mlp(train, val, with_lags, max_epochs=epochs, batch_size=batch)
    out["mlp_q"] = both(mq)
    intervals["mlp_q"] = mq.predict_quantiles(test)
    return out, intervals


def run_model_zoo(series: pd.DataFrame, weather_fc: pd.DataFrame, weather_provider: pd.DataFrame | None = None,
                  horizons=(15, 30, 60), bin_minutes: int = 15, n_lags: int = 12, seeds=(0, 1), epochs: int = 40,
                  batch_size: int = 1024, tz: str = "UTC", placebo_seed: int = 0):
    """Return (results, comparisons, intervals, notes).

    Comparisons include `forecast_vs_placebo`: real archived-forecast weather against the same
    weather shuffled across days. That is the direct test of whether weather carries signal.
    """
    labels = series["data_label"].unique()
    if len(labels) != 1:
        raise ValueError(f"series must have exactly one data_label, got {sorted(labels)}")
    label, lag_cols = labels[0], lag_columns(n_lags)
    sets = {"traffic": list(TRAFFIC_COLS), "traffic+forecast": TRAFFIC_COLS + WEATHER_COLS["forecast"],
            "traffic+placebo": TRAFFIC_COLS + WEATHER_COLS["placebo"]}
    placebo = shuffle_days(weather_fc, seed=placebo_seed, tz=tz)
    if weather_provider is not None:
        sets["traffic+provider"] = TRAFFIC_COLS + WEATHER_COLS["provider"]

    rows, comps, ivals, notes = [], [], [], []
    for h in horizons:
        ex = build_examples(series, h, bin_minutes, tz=tz, n_lags=n_lags)
        ex = add_weather(ex, weather_fc, "forecast")
        ex = add_weather(ex, placebo, "placebo")
        if weather_provider is not None:
            ex = add_weather(ex, weather_provider, "provider")
        folds = chronological_split(ex, gap_min=max(horizons))
        train, val, test = folds["train"], folds["val"], folds["test"]
        y_val, y_test = val["target_duration_s"].to_numpy(), test["target_duration_s"].to_numpy()
        blocks = test["target_bin_start"].dt.date.to_numpy()
        rainy = int(test["wx_fc_rain"].sum())
        if len(set(blocks)) < 20:
            notes.append(f"horizon {h}: only {len(set(blocks))} test days; bootstrap intervals are unreliable")
        if rainy < 100:
            notes.append(f"horizon {h}: only {rainy} rainy test examples; the weather comparison has little power")

        preds = {("persistence", "last_obs"): (predict_persistence(val).to_numpy(), predict_persistence(test).to_numpy()),
                 ("historical_mean", "calendar"): (HistoricalMean.fit(train).predict(val).to_numpy(),
                                                   HistoricalMean.fit(train).predict(test).to_numpy())}
        interval_store = {}
        for name, cols in sets.items():
            fitted, interval = _fit_models(train, val, test, cols, lag_cols, seeds, epochs, batch_size)
            for model, p in fitted.items():
                preds[(model, name)] = p
            for model, q in interval.items():
                interval_store[(model, name)] = q

        persist_mae = mae(y_test, preds[("persistence", "last_obs")][1])
        for (model, fs), (pv, pt) in preds.items():
            rows.append({"horizon_min": h, "model": model, "feature_set": fs, "n_test": len(test), "test_rainy": rainy,
                         "val_mae_s": mae(y_val, pv), "mae_s": mae(y_test, pt), "rmse_s": rmse(y_test, pt),
                         "vs_persistence_pct": 100 * (1 - mae(y_test, pt) / persist_mae), "data_label": label})
        for (model, fs), q in interval_store.items():
            lo, hi = q[:, 0], q[:, -1]
            ivals.append({"horizon_min": h, "model": model, "feature_set": fs, "nominal": 0.8,
                          "coverage": interval_coverage(y_test, lo, hi), "mean_width_s": mean_interval_width(lo, hi),
                          "data_label": label})
        for model in MODELS:
            for fs in sets:
                if fs == "traffic":
                    continue
                r = paired_mae_difference(y_test, preds[(model, fs)][1], preds[(model, "traffic")][1], blocks)
                comps.append({"horizon_min": h, "model": model, "with": fs, "mean_diff_s": r["mean_diff"],
                              "ci_low_s": r["ci_low"], "ci_high_s": r["ci_high"], "n": r["n"], "n_days": r["n_blocks"],
                              "verdict": _verdict(r["ci_low"], r["ci_high"]), "data_label": label})
            r = paired_mae_difference(y_test, preds[(model, "traffic+forecast")][1],
                                      preds[(model, "traffic+placebo")][1], blocks)
            comps.append({"horizon_min": h, "model": model, "with": "forecast_vs_placebo",
                          "mean_diff_s": r["mean_diff"], "ci_low_s": r["ci_low"], "ci_high_s": r["ci_high"],
                          "n": r["n"], "n_days": r["n_blocks"], "verdict": _verdict(r["ci_low"], r["ci_high"]),
                          "data_label": label})

    results = pd.DataFrame(rows)
    trad = results[(results.feature_set == "traffic") & results.model.isin(MODELS)]
    best = trad.loc[trad.groupby("horizon_min")["val_mae_s"].idxmin()][["horizon_min", "model"]]
    results = results.merge(best.assign(selected_by_validation=True), on=["horizon_min", "model"], how="left")
    results["selected_by_validation"] = (results["selected_by_validation"].fillna(False).astype(bool)
                                         & (results["feature_set"] == "traffic"))
    notes.append(f"{len(comps)} weather comparisons at 95% intervals: about {max(1, round(0.05 * len(comps)))} "
                 "'significant' result(s) are expected by chance alone")
    return results, pd.DataFrame(comps), pd.DataFrame(ivals), notes
