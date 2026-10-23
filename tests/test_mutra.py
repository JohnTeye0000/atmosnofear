"""Loader tests with a tiny hand-made CSV in MUTra's column layout (a fixture, not the real data)."""
import pandas as pd
import pytest

from atmosnofear import mutra
from atmosnofear.features import add_weather, build_examples

COLS = ["segment_id", "datetime", "current_speed", "free_flow_speed", "current_travel_time",
        "free_flow_travel_time", "temperature_c", "precipitation_hourly_total_mm", "wind_speed_ms",
        "relative_speed_index", "congestion_index_0_10", "is_imputed"]


def make_csv(tmp_path, rows):
    df = pd.DataFrame(rows, columns=["segment_id", "datetime", "tt", "imputed", "precip"])
    full = pd.DataFrame({"segment_id": df.segment_id, "datetime": df.datetime, "current_speed": 20,
                         "free_flow_speed": 25, "current_travel_time": df.tt, "free_flow_travel_time": 60,
                         "temperature_c": 18.0, "precipitation_hourly_total_mm": df.precip, "wind_speed_ms": 2.0,
                         "relative_speed_index": 0.8, "congestion_index_0_10": 2.0, "is_imputed": df.imputed})[COLS]
    path = tmp_path / "m.csv"
    full.to_csv(path, index=False)
    return path


ROWS = [(1, "2024-11-01 08:00:00", 60, False, 0.0), (1, "2024-11-01 08:05:00", 90, False, 0.0),
        (1, "2024-11-01 08:10:00", 120, False, 0.0), (1, "2024-11-01 08:15:00", 999, True, 0.0),
        (1, "2024-11-01 09:00:00", 70, False, 1.5)]


def test_load_series_converts_local_time_to_utc_bins_and_labels(tmp_path):
    s = mutra.load_series(make_csv(tmp_path, ROWS))
    first = s.iloc[0]
    assert first["bin_start_utc"] == pd.Timestamp("2024-11-01T14:00:00Z")   # 08:00 CDMX = 14:00 UTC
    assert first["median_duration_s"] == 90 and first["n_obs"] == 3        # median of 60, 90, 120
    assert set(s["data_label"]) == {"provider_estimate"} and s["segment_id"].iloc[0] == "1"


def test_imputed_rows_are_dropped_by_default(tmp_path):
    path = make_csv(tmp_path, ROWS)
    assert 999 not in set(mutra.load_series(path)["median_duration_s"])
    assert mutra.load_series(path, include_imputed=True)["n_obs"].sum() == 5


def test_provider_weather_is_hourly_utc_and_labelled(tmp_path):
    w = mutra.load_provider_weather(make_csv(tmp_path, ROWS))
    assert len(w) == 2 and set(w["data_label"]) == {"provider_weather"}
    assert w["valid_time_utc"].iloc[1] == pd.Timestamp("2024-11-01T15:00:00Z")
    assert w["precipitation"].iloc[1] == 1.5


def test_provider_weather_only_accepted_in_provider_mode(tmp_path):
    w = mutra.load_provider_weather(make_csv(tmp_path, ROWS))
    ex = build_examples(mutra.load_series(make_csv(tmp_path, ROWS), 5), 5, 5)
    assert "wx_prov_precipitation" in add_weather(ex, w, "provider").columns
    for wrong in ("forecast", "oracle"):
        with pytest.raises(ValueError):
            add_weather(ex, w, wrong)


def test_calendar_features_use_local_time():
    idx = pd.date_range("2024-11-01T14:00:00Z", periods=6, freq="15min")   # 08:00 local
    series = pd.DataFrame({"segment_id": "1", "bin_start_utc": idx, "median_duration_s": 60.0,
                           "n_obs": 1, "data_label": "provider_estimate"})
    utc = build_examples(series, 15, 15)
    local = build_examples(series, 15, 15, tz="America/Mexico_City")
    assert utc["tod_min"].iloc[0] == 14 * 60 + 15 and local["tod_min"].iloc[0] == 8 * 60 + 15
    pd.testing.assert_series_equal(utc["target_duration_s"], local["target_duration_s"])


def test_late_evening_local_time_stays_on_the_same_weekday():
    # 20:00 local on Friday 2024-11-01 is 02:00 UTC on Saturday: weekday must follow local time
    idx = pd.date_range("2024-11-02T02:00:00Z", periods=4, freq="15min")
    series = pd.DataFrame({"segment_id": "1", "bin_start_utc": idx, "median_duration_s": 60.0,
                           "n_obs": 1, "data_label": "provider_estimate"})
    assert build_examples(series, 15, 15, tz="America/Mexico_City")["dow"].iloc[0] == 4   # Friday
    assert build_examples(series, 15, 15)["dow"].iloc[0] == 5                             # Saturday in UTC
