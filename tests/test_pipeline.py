"""Tests for binning, features, split, baselines, metrics and evaluation.

All trip data here is SYNTHETIC and exists only to test the logic. Nothing asserts
that a model is accurate; tests check correctness and leakage, not performance.
"""
import numpy as np
import pandas as pd
import pytest

from atmosnofear import synthetic, trips
from atmosnofear.baselines import HistoricalMean, predict_persistence
from atmosnofear.binning import bin_trips
from atmosnofear.evaluate import evaluate_baselines
from atmosnofear.features import add_weather, build_examples
from atmosnofear.metrics import mae, paired_mae_difference, rmse
from atmosnofear.split import chronological_split


def make_series(values, start="2025-01-06T08:00:00Z", seg="S1", label="synthetic"):
    """A regular 15-minute series; None means an empty bin."""
    idx = pd.date_range(start, periods=len(values), freq="15min")
    df = pd.DataFrame({"segment_id": seg, "bin_start_utc": idx, "median_duration_s": values,
                       "n_obs": 1, "data_label": label})
    return df.dropna(subset=["median_duration_s"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def syn_series():
    return bin_trips(synthetic.generate_trips(days=28, seed=1))


# ---- binning
def test_bin_takes_median_and_counts():
    t = pd.DataFrame({"segment_id": "S1", "data_label": "observed",
                      "start_time_utc": ["2025-01-06T08:01:00Z", "2025-01-06T08:10:00Z", "2025-01-06T08:16:00Z"],
                      "duration_s": [100.0, 300.0, 500.0]})
    out = bin_trips(t, 15)
    assert list(out["median_duration_s"]) == [200.0, 500.0] and list(out["n_obs"]) == [2, 1]


def test_bin_refuses_mixed_labels():
    t = pd.DataFrame({"segment_id": "S1", "data_label": ["observed", "synthetic"],
                      "start_time_utc": ["2025-01-06T08:00:00Z"] * 2, "duration_s": [1.0, 2.0]})
    with pytest.raises(ValueError):
        bin_trips(t)


# ---- features and leakage
def test_target_alignment_for_each_horizon():
    s = make_series([float(v) for v in range(10, 20)])  # 10..19, one per bin
    ex15 = build_examples(s, 15)
    first = ex15.iloc[0]
    assert first["last_obs_s"] == 10 and first["target_duration_s"] == 11   # next bin
    assert first["t_utc"] == pd.Timestamp("2025-01-06T08:15:00Z")           # known at end of bin
    assert build_examples(s, 30).iloc[0]["target_duration_s"] == 12


def test_features_ignore_the_future():
    """Changing a value after the prediction time must not change that row's features."""
    base = [float(v) for v in range(10, 30)]
    changed = base.copy()
    changed[12:] = [999.0] * len(changed[12:])
    a, b = build_examples(make_series(base), 15), build_examples(make_series(changed), 15)
    cols = ["last_obs_s", "mean4_s", "stale_bins"]
    # rows whose source bin index <= 10 only see bins <= 10
    pd.testing.assert_frame_equal(a.loc[:10, cols], b.loc[:10, cols])


def test_stale_bins_and_staleness_cap():
    vals = [10.0, 12.0] + [None] * 11 + [20.0]     # 11 empty bins, then a value at index 13
    ex = build_examples(make_series(vals), 15)
    assert list(ex["target_duration_s"]) == [12.0]  # only the first row has a known last value and target
    assert (ex["stale_bins"] <= 8).all()            # the row targeting 20.0 is stale (12 bins) and dropped


def test_horizon_must_be_multiple_of_bin():
    with pytest.raises(ValueError):
        build_examples(make_series([1.0, 2.0, 3.0]), 20)


def _weather(label, precip):
    t = pd.date_range("2025-01-06T08:00:00Z", periods=len(precip), freq="h")
    return pd.DataFrame({"valid_time_utc": t, "precipitation": precip, "temperature_2m": 25.0, "data_label": label})


def test_add_weather_forecast_rejects_reanalysis_labelled_data():
    ex = build_examples(make_series([float(v) for v in range(10, 20)]), 15)
    with pytest.raises(ValueError):
        add_weather(ex, _weather("reanalysis", [0.0] * 5), "forecast")
    with pytest.raises(ValueError):
        add_weather(ex, _weather("nwp_forecast", [0.0] * 5), "oracle")


def test_add_weather_matches_target_hour_and_keeps_missing_as_nan():
    ex = build_examples(make_series([float(v) for v in range(10, 30)]), 15)
    out = add_weather(ex, _weather("nwp_forecast", [0.0, 2.0]), "forecast")  # only 08:00 and 09:00 known
    assert len(out) == len(ex)
    row = out[out["target_bin_start"] == pd.Timestamp("2025-01-06T09:00:00Z")].iloc[0]
    assert row["wx_fc_precipitation"] == 2.0 and row["wx_fc_rain"] == 1.0
    assert out["wx_fc_rain"].isna().any()          # hours outside the weather range stay NaN


# ---- split
def test_split_is_ordered_non_overlapping_and_gapped(syn_series):
    ex = build_examples(syn_series, 60)
    f = chronological_split(ex, gap_min=60)
    assert f["train"]["t_utc"].max() < f["val"]["t_utc"].min()
    assert f["val"]["t_utc"].max() < f["test"]["t_utc"].min()
    assert f["val"]["t_utc"].min() - f["train"]["t_utc"].max() >= pd.Timedelta(minutes=60)
    assert sum(len(v) for v in f.values()) <= len(ex)


def test_split_rejects_bad_fractions(syn_series):
    with pytest.raises(ValueError):
        chronological_split(build_examples(syn_series, 15), train_frac=0.9, val_frac=0.2)


# ---- baselines
def test_historical_mean_uses_only_training_data_and_falls_back():
    tr = pd.DataFrame({"segment_id": "S1", "dow": 0, "tod_min": 480, "target_duration_s": [100.0] * 3})
    model = HistoricalMean.fit(tr)
    ex = pd.DataFrame({"segment_id": ["S1", "S1", "OTHER"], "dow": [0, 3, 0], "tod_min": [480, 480, 480]})
    assert list(model.predict(ex)) == [100.0, 100.0, 100.0]  # cell, segment fallback, overall fallback


def test_persistence_returns_last_observation():
    ex = pd.DataFrame({"last_obs_s": [5, 7]})
    assert list(predict_persistence(ex)) == [5.0, 7.0]


# ---- metrics
def test_mae_rmse_known_values():
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_bootstrap_detects_a_clear_difference_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = rng.normal(100, 5, 600)
    good, bad = y + rng.normal(0, 1, 600), y + rng.normal(0, 8, 600)
    blocks = np.repeat(np.arange(30), 20)
    r1 = paired_mae_difference(y, good, bad, blocks, seed=3)
    r2 = paired_mae_difference(y, good, bad, blocks, seed=3)
    assert r1 == r2 and r1["ci_high"] < 0          # A is clearly better; interval excludes 0


def test_bootstrap_interval_covers_zero_for_identical_models():
    y = np.random.default_rng(1).normal(100, 5, 400)
    r = paired_mae_difference(y, y + 1, y + 1, np.repeat(np.arange(20), 20))
    assert r["ci_low"] <= 0 <= r["ci_high"]


# ---- end to end
def test_evaluate_baselines_runs_and_carries_data_label(syn_series):
    out = evaluate_baselines(syn_series)
    assert set(out["data_label"]) == {"synthetic"}
    assert set(out["horizon_min"]) == {15, 30, 60} and set(out["model"]) == {"historical_mean", "persistence"}
    assert out[["mae_min", "rmse_min"]].notna().all().all()
    assert (out["rmse_min"] >= out["mae_min"]).all()


def test_synthetic_trips_pass_validation():
    clean, issues = trips.validate_trips(synthetic.generate_trips(days=3), synthetic.synthetic_segments())
    assert issues == [] and len(clean) == 120


def test_evaluate_rejects_mixed_labels():
    mixed = pd.concat([make_series([1.0] * 20, label="observed"), make_series([1.0] * 20, seg="S2")])
    with pytest.raises(ValueError):
        evaluate_baselines(mixed)
