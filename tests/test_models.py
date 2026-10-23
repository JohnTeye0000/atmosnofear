"""Model and experiment tests. Traffic here is SYNTHETIC and weather is a made-up fixture.

These tests check that the learners work, are reproducible, do not use test-fold
statistics, and that the experiment is wired correctly. They do not claim accuracy.
"""
import numpy as np
import pandas as pd
import pytest

from atmosnofear import synthetic
from atmosnofear.binning import bin_trips
from atmosnofear.experiment import run_weather_experiment
from atmosnofear.metrics import mae
from atmosnofear.models import Encoder, TRAFFIC_COLS, fit_gbm, fit_mlp


def learnable(n=600, seed=0, seg="S1"):
    """Examples where target = 1.5 * last_obs_s + noise, so a working learner must beat the mean."""
    rng = np.random.default_rng(seed)
    last = rng.uniform(300, 900, n)
    df = pd.DataFrame({"segment_id": seg, "last_obs_s": last, "mean4_s": last + rng.normal(0, 5, n),
                       "stale_bins": rng.integers(0, 3, n), "tod_min": rng.integers(360, 1200, n),
                       "dow": rng.integers(0, 7, n), "is_weekend": rng.integers(0, 2, n)})
    return df.assign(target_duration_s=1.5 * last + rng.normal(0, 10, n))


# ---- encoder
def test_encoder_uses_only_training_statistics():
    train, test = learnable(seed=1), learnable(seed=2).assign(last_obs_s=1e6)   # test is wildly different
    enc = Encoder.fit(train, TRAFFIC_COLS)
    assert enc.means[0] == pytest.approx(train["last_obs_s"].mean())            # unaffected by test rows
    assert enc.transform(test, impute=True, scale=True).shape[0] == len(test)


def test_encoder_nan_kept_for_boosting_and_imputed_with_flag_for_network():
    train = learnable().assign(wx=lambda d: np.where(d.index % 5 == 0, np.nan, 1.0))
    enc = Encoder.fit(train, TRAFFIC_COLS + ["wx"])
    assert np.isnan(enc.transform(train, impute=False, scale=False)).any()
    imputed = enc.transform(train, impute=True, scale=True)
    assert not np.isnan(imputed).any()
    assert imputed.shape[1] == len(TRAFFIC_COLS) + 1 + 1 + 1       # cols + wx + segment one-hot + missing flag


def test_unseen_segment_encodes_as_zero_onehot():
    enc = Encoder.fit(learnable(seg="A"), TRAFFIC_COLS)
    x = enc.transform(learnable(seg="B"), impute=False, scale=False)
    assert (x[:, -1] == 0).all()


# ---- learners
def test_gbm_learns_a_simple_relationship():
    tr, va, te = learnable(seed=1), learnable(200, seed=2), learnable(200, seed=3)
    pred = fit_gbm(tr, va, TRAFFIC_COLS).predict(te)
    naive = np.full(len(te), tr["target_duration_s"].mean())
    assert mae(te["target_duration_s"], pred) < 0.3 * mae(te["target_duration_s"], naive)


def test_mlp_learns_a_simple_relationship_and_is_reproducible():
    tr, va, te = learnable(seed=1), learnable(200, seed=2), learnable(200, seed=3)
    a = fit_mlp(tr, va, TRAFFIC_COLS, seed=0, max_epochs=60).predict(te)
    b = fit_mlp(tr, va, TRAFFIC_COLS, seed=0, max_epochs=60).predict(te)
    c = fit_mlp(tr, va, TRAFFIC_COLS, seed=1, max_epochs=60).predict(te)
    naive = np.full(len(te), tr["target_duration_s"].mean())
    assert mae(te["target_duration_s"], a) < 0.5 * mae(te["target_duration_s"], naive)
    np.testing.assert_array_equal(a, b)            # same seed, same result
    assert not np.array_equal(a, c)                # different seed, different network


def test_empty_validation_fold_is_rejected():
    tr = learnable()
    with pytest.raises(ValueError):
        fit_gbm(tr, tr.iloc[0:0], TRAFFIC_COLS)
    with pytest.raises(ValueError):
        fit_mlp(tr, tr.iloc[0:0], TRAFFIC_COLS)


# ---- experiment
def weather(label, seed, days=35):
    t = pd.date_range("2025-01-05T00:00:00Z", periods=days * 24, freq="h")
    rng = np.random.default_rng(seed)
    rain = np.where(rng.random(len(t)) < 0.25, rng.exponential(2.0, len(t)), 0.0)
    return pd.DataFrame({"valid_time_utc": t, "precipitation": rain,
                         "temperature_2m": 25 + rng.normal(0, 2, len(t)), "data_label": label})


@pytest.fixture(scope="module")
def experiment():
    series = bin_trips(synthetic.generate_trips(days=28, seed=5))
    return run_weather_experiment(series, weather("nwp_forecast", 1), weather("reanalysis", 2),
                                  horizons=(15, 30), seeds=(0, 1), mlp_epochs=25)


def test_experiment_covers_all_models_and_feature_sets(experiment):
    results, comps, _ = experiment
    assert set(results["model"]) == {"historical_mean", "persistence", "gbm", "mlp"}
    assert set(results["feature_set"]) >= {"traffic", "traffic+forecast", "traffic+oracle"}
    assert set(results["data_label"]) == {"synthetic"} and set(comps["data_label"]) == {"synthetic"}
    assert results[["mae_min", "rmse_min"]].notna().all().all()


def test_experiment_comparisons_have_valid_verdicts(experiment):
    _, comps, _ = experiment
    assert len(comps) == 2 * 2 * 2                    # horizons x models x weather sets
    assert set(comps["verdict"]) <= {"weather lowers error", "weather raises error", "no difference detected"}
    assert (comps["ci_low_min"] <= comps["ci_high_min"]).all()


def test_negative_control_synthetic_traffic_has_no_weather_effect(experiment):
    """Traffic was generated without any weather dependence, so weather must not help much.

    A large 'improvement' here would indicate leakage or a bug. We allow a small tolerance
    because random weather can be fit by chance.
    """
    results, _, _ = experiment
    pivot = results[results.model.isin(["gbm", "mlp"])].pivot_table(
        index=["horizon_min", "model"], columns="feature_set", values="mae_min")
    gain = (pivot["traffic"] - pivot["traffic+forecast"]) / pivot["traffic"]
    assert (gain < 0.10).all(), gain


def test_experiment_reports_rain_counts_and_power_notes(experiment):
    results, _, notes = experiment
    assert results["test_rainy"].notna().all()
    if (results["test_rainy"] < 100).any():
        assert any("rainy test examples" in n for n in notes)
    assert any("test days" in n for n in notes)            # 28 synthetic days -> few test days
    assert any("expected by chance" in n for n in notes)   # multiple-comparison caution is always given


def test_experiment_refuses_mislabelled_weather():
    series = bin_trips(synthetic.generate_trips(days=14, seed=5))
    with pytest.raises(ValueError):
        run_weather_experiment(series, weather("reanalysis", 1), horizons=(15,), seeds=(0,), mlp_epochs=2)


def test_experiment_traffic_only_when_no_weather_given():
    series = bin_trips(synthetic.generate_trips(days=14, seed=5))
    results, comps, _ = run_weather_experiment(series, horizons=(15,), seeds=(0,), mlp_epochs=3)
    assert comps.empty and set(results["feature_set"]) == {"calendar", "last_obs", "traffic"}
