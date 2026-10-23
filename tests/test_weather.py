"""Offline tests: no network. Payloads here are synthetic fixtures for logic only."""
from datetime import date, datetime, timezone

import pytest

from atmosnofear import weather

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def payload(times, temps):
    return {"latitude": 6.71, "longitude": -1.59,
            "hourly": {"time": times, "temperature_2m": temps}}


def test_parse_labels_forecast_and_reanalysis_differently():
    p = payload(["2024-09-01T00:00"], [22.0])
    f = weather.parse_hourly(p, "forecast", NOW)
    r = weather.parse_hourly(p, "reanalysis", NOW)
    assert f["data_label"].iloc[0] == "nwp_forecast"
    assert r["data_label"].iloc[0] == "reanalysis"
    assert str(f["valid_time_utc"].dt.tz) == "UTC"


def test_parse_keeps_grid_point():
    f = weather.parse_hourly(payload(["2024-09-01T00:00"], [22.0]), "forecast", NOW)
    assert f["grid_lat"].iloc[0] == 6.71


def test_parse_rejects_error_payload():
    with pytest.raises(ValueError):
        weather.parse_hourly({"error": True, "reason": "bad"}, "forecast", NOW)


def test_merge_prefers_fresh_rows_and_sorts():
    old = weather.parse_hourly(payload(["2024-09-01T00:00", "2024-09-01T01:00"], [20.0, 21.0]), "forecast", NOW)
    new = weather.parse_hourly(payload(["2024-09-01T01:00", "2024-09-01T02:00"], [99.0, 23.0]), "forecast", NOW)
    merged = weather.merge_cache(old, new)
    assert list(merged["temperature_2m"]) == [20.0, 99.0, 23.0]
    assert merged["valid_time_utc"].is_monotonic_increasing


def test_merge_does_not_mutate_inputs():
    old = weather.parse_hourly(payload(["2024-09-01T00:00"], [20.0]), "forecast", NOW)
    new = weather.parse_hourly(payload(["2024-09-01T01:00"], [21.0]), "forecast", NOW)
    weather.merge_cache(old, new)
    assert len(old) == 1 and len(new) == 1


def test_next_start_empty_cache_uses_default():
    assert weather.next_start(None, date(2024, 1, 1)) == date(2024, 1, 1)


def test_next_start_resumes_with_overlap():
    cached = weather.parse_hourly(payload(["2024-09-10T05:00"], [20.0]), "forecast", NOW)
    assert weather.next_start(cached, date(2024, 1, 1)) == date(2024, 9, 9)


def test_forecast_requires_explicit_model():
    with pytest.raises(ValueError):
        weather.fetch("forecast", 6.7, -1.6, date(2024, 1, 1), date(2024, 1, 2), ["temperature_2m"])


def test_parse_strips_model_suffix_and_records_model():
    p = {"latitude": 6.7, "longitude": -1.6,
         "hourly": {"time": ["2024-09-01T00:00"], "temperature_2m_ecmwf_ifs025": [22.0]}}
    df = weather.parse_hourly(p, "forecast", NOW, "ecmwf_ifs025")
    assert "temperature_2m" in df.columns and df["model"].iloc[0] == "ecmwf_ifs025"
