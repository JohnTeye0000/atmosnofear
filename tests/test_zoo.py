"""Tests for lag features, residual targets, GRU, quantile models and the zoo harness.

All data is SYNTHETIC or a hand-made fixture. Tests check correctness, shapes, leakage and
reproducibility; they do not claim real-world accuracy.
"""
import numpy as np
import pandas as pd
import pytest

from atmosnofear import synthetic
from atmosnofear.binning import bin_trips
from atmosnofear.deep import fit_gru, fit_quantile_mlp, pinball_loss
from atmosnofear.features import build_examples, lag_columns
from atmosnofear.metrics import interval_coverage, mae, mean_interval_width
from atmosnofear.models import TRAFFIC_COLS, fit_gbm, fit_gbm_quantiles, fit_mlp
from atmosnofear.zoo import MODELS, run_model_zoo, shuffle_days

LAGS = lag_columns(6)


def learnable(n=900, seed=0):
    """target = 1.1 * last + noise; lags are noisy copies of last, so history is informative."""
    rng = np.random.default_rng(seed)
    last = rng.uniform(40, 200, n)
    df = pd.DataFrame({"segment_id": "S1", "last_obs_s": last, "mean4_s": last + rng.normal(0, 3, n),
                       "stale_bins": rng.integers(0, 3, n), "tod_min": rng.integers(360, 1200, n),
                       "dow": rng.integers(0, 7, n), "is_weekend": rng.integers(0, 2, n)})
    for j, name in enumerate(LAGS):
        df[name] = np.where(rng.random(n) < 0.1, np.nan, last + rng.normal(0, 5, n)) if j else last
    return df.assign(target_duration_s=1.1 * last + rng.normal(0, 3, n))


# ---- lag features
def test_lag_features_follow_history_and_ignore_the_future():
    idx = pd.date_range("2025-01-06T08:00:00Z", periods=20, freq="15min")
    base = pd.DataFrame({"segment_id": "S1", "bin_start_utc": idx, "median_duration_s": np.arange(10.0, 30.0),
                         "n_obs": 1, "data_label": "synthetic"})
    changed = base.assign(median_duration_s=lambda d: np.where(np.arange(len(d)) >= 12, 999.0, d.median_duration_s))
    a, b = build_examples(base, 15, 15, n_lags=4), build_examples(changed, 15, 15, n_lags=4)
    assert list(a.loc[3, lag_columns(4)]) == [13.0, 12.0, 11.0, 10.0]         # lag_0 is the newest bin
    pd.testing.assert_frame_equal(a.loc[:10, lag_columns(4)], b.loc[:10, lag_columns(4)])


def test_no_lag_columns_by_default():
    idx = pd.date_range("2025-01-06T08:00:00Z", periods=6, freq="15min")
    s = pd.DataFrame({"segment_id": "S1", "bin_start_utc": idx, "median_duration_s": 1.0, "n_obs": 1,
                      "data_label": "synthetic"})
    assert not any(c.startswith("lag_") for c in build_examples(s, 15, 15).columns)


# ---- residual targets
def test_residual_models_learn_and_predictions_are_on_the_original_scale():
    tr, va, te = learnable(seed=1), learnable(300, seed=2), learnable(300, seed=3)
    cols = TRAFFIC_COLS + LAGS
    naive = mae(te["target_duration_s"], np.full(len(te), tr["target_duration_s"].mean()))
    g = fit_gbm(tr, va, cols, residual=True)
    m = fit_mlp(tr, va, cols, max_epochs=40, batch_size=128, patience=5, residual=True)
    for p in (g.predict(te), m.predict(te)):
        assert mae(te["target_duration_s"], p) < 0.25 * naive
    assert g.residual and m.residual


# ---- GRU
def test_gru_learns_is_reproducible_and_needs_lags():
    tr, va, te = learnable(seed=1), learnable(300, seed=2), learnable(300, seed=3)
    a = fit_gru(tr, va, TRAFFIC_COLS, LAGS, seed=0, max_epochs=15).predict(te)
    b = fit_gru(tr, va, TRAFFIC_COLS, LAGS, seed=0, max_epochs=15).predict(te)
    naive = mae(te["target_duration_s"], np.full(len(te), tr["target_duration_s"].mean()))
    assert mae(te["target_duration_s"], a) < 0.5 * naive
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        fit_gru(tr, va, TRAFFIC_COLS, [])
    with pytest.raises(ValueError):
        fit_gru(tr, va.iloc[0:0], TRAFFIC_COLS, LAGS)


def test_gru_handles_missing_lags():
    tr, va = learnable(seed=1), learnable(200, seed=2)
    te = learnable(100, seed=3).assign(**{c: np.nan for c in LAGS[1:]})
    out = fit_gru(tr, va, TRAFFIC_COLS, LAGS, max_epochs=3).predict(te)
    assert np.isfinite(out).all() and len(out) == 100


# ---- quantiles
def test_pinball_loss_known_values():
    loss = pinball_loss((0.5,))
    import torch
    assert float(loss(torch.tensor([[1.0]]), torch.tensor([3.0]))) == pytest.approx(1.0)   # 0.5 * |3 - 1|
    assert float(pinball_loss((0.9,))(torch.tensor([[1.0]]), torch.tensor([3.0]))) == pytest.approx(1.8)
    assert float(pinball_loss((0.9,))(torch.tensor([[3.0]]), torch.tensor([1.0]))) == pytest.approx(0.2)


@pytest.mark.parametrize("kind", ["mlp", "gbm"])
def test_quantiles_are_ordered_and_roughly_calibrated(kind):
    tr, va, te = learnable(1500, seed=1), learnable(300, seed=2), learnable(1500, seed=3)
    cols = TRAFFIC_COLS + LAGS
    if kind == "mlp":
        model = fit_quantile_mlp(tr, va, cols, max_epochs=40, batch_size=128, patience=6)
    else:
        model = fit_gbm_quantiles(tr, cols, {"max_iter": 150, "max_leaf_nodes": 8, "learning_rate": 0.05})
    q = model.predict_quantiles(te)
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()
    cov = interval_coverage(te["target_duration_s"], q[:, 0], q[:, 2])
    assert 0.55 < cov < 0.97                       # nominal 0.80; loose bounds for a small test
    assert mean_interval_width(q[:, 0], q[:, 2]) > 0
    assert model.predict(te).shape == (len(te),)


def test_quantile_requires_median():
    tr, va = learnable(), learnable(200, seed=2)
    with pytest.raises(ValueError):
        fit_quantile_mlp(tr, va, TRAFFIC_COLS, quantiles=(0.1, 0.9))


# ---- zoo
def weather(label, seed, days=35):
    t = pd.date_range("2025-01-05T00:00:00Z", periods=days * 24, freq="h")
    rng = np.random.default_rng(seed)
    rain = np.where(rng.random(len(t)) < 0.25, rng.exponential(2.0, len(t)), 0.0)
    return pd.DataFrame({"valid_time_utc": t, "precipitation": rain,
                         "temperature_2m": 25 + rng.normal(0, 2, len(t)), "data_label": label})


@pytest.fixture(scope="module")
def zoo():
    series = bin_trips(synthetic.generate_trips(days=28, seed=5))
    return run_model_zoo(series, weather("nwp_forecast", 1), weather("provider_weather", 3),
                         horizons=(15, 30), seeds=(0,), epochs=6, batch_size=256, n_lags=6)


def test_zoo_covers_every_model_and_feature_set(zoo):
    results, comps, intervals, _ = zoo
    assert set(MODELS) <= set(results["model"]) and {"persistence", "historical_mean"} <= set(results["model"])
    assert set(results["feature_set"]) >= {"traffic", "traffic+forecast", "traffic+provider", "traffic+placebo"}
    assert set(results["data_label"]) == {"synthetic"}
    assert results[["mae_s", "rmse_s", "val_mae_s"]].notna().all().all()
    # per model and horizon: forecast, placebo, provider vs traffic, plus forecast vs placebo
    assert len(comps) == len(MODELS) * 4 * 2
    assert set(comps["with"]) == {"traffic+forecast", "traffic+placebo", "traffic+provider", "forecast_vs_placebo"}


def test_shuffle_days_preserves_values_but_breaks_the_date_link():
    t = pd.date_range("2025-01-05T00:00:00Z", periods=10 * 24, freq="h")
    day = np.repeat(np.arange(10), 24)
    w = pd.DataFrame({"valid_time_utc": t, "precipitation": day * 1.0, "temperature_2m": 20.0 + day,
                      "data_label": "nwp_forecast"})
    p = shuffle_days(w, seed=3)
    assert set(p["data_label"]) == {"placebo_weather"} and len(p) == len(w)
    assert (p["valid_time_utc"] == w["valid_time_utc"]).all()
    assert sorted(p["precipitation"].unique()) == sorted(w["precipitation"].unique())   # same day-values reused
    assert (p["precipitation"] != w["precipitation"]).any()                              # but moved to other days
    assert (p.groupby(day)["precipitation"].nunique() == 1).all()                        # a day stays one block
    pd.testing.assert_frame_equal(p, shuffle_days(w, seed=3))                            # deterministic
    assert not p["precipitation"].equals(shuffle_days(w, seed=4)["precipitation"])


def test_zoo_selects_exactly_one_traffic_model_per_horizon_by_validation(zoo):
    results, _, _, _ = zoo
    chosen = results[results["selected_by_validation"]]
    assert len(chosen) == 2 and set(chosen["feature_set"]) == {"traffic"}
    for h, row in chosen.set_index("horizon_min").iterrows():
        pool = results[(results.horizon_min == h) & (results.feature_set == "traffic") & results.model.isin(MODELS)]
        assert row["val_mae_s"] == pool["val_mae_s"].min()      # chosen on validation, not test


def test_zoo_intervals_are_valid_probabilities(zoo):
    _, _, intervals, _ = zoo
    assert set(intervals["model"]) == {"gbm_q", "mlp_q"}
    assert intervals["coverage"].between(0, 1).all() and (intervals["mean_width_s"] > 0).all()


def test_zoo_negative_control_weather_does_not_help_much(zoo):
    results, _, _, _ = zoo
    piv = results[results.model.isin(MODELS)].pivot_table(index=["horizon_min", "model"], columns="feature_set",
                                                          values="mae_s")
    gain = (piv["traffic"] - piv["traffic+forecast"]) / piv["traffic"]
    assert (gain < 0.15).all(), gain             # synthetic traffic has no weather dependence


def test_zoo_refuses_mislabelled_weather():
    series = bin_trips(synthetic.generate_trips(days=14, seed=5))
    with pytest.raises(ValueError):
        run_model_zoo(series, weather("reanalysis", 1), horizons=(15,), seeds=(0,), epochs=1, n_lags=4)
