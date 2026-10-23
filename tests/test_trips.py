"""Offline tests. Rows are synthetic fixtures used only to test validation logic."""
import pandas as pd

from atmosnofear import trips

SEG = pd.DataFrame({"segment_id": ["S1"], "length_m": [3600.0]})  # 3.6 km


def row(**over):
    base = {"trip_id": "t1", "segment_id": "S1", "start_time_utc": "2025-01-01T08:00:00Z",
            "duration_s": 600.0, "vehicle_type": "trotro", "collector_id": "P001",
            "data_label": "observed"}
    return {**base, **over}


def validate(*rows):
    return trips.validate_trips(pd.DataFrame(rows), SEG)


def test_valid_row_passes():
    clean, issues = validate(row())
    assert len(clean) == 1 and issues == []


def test_rejects_implausible_speed():
    clean, issues = validate(row(duration_s=5.0))
    assert clean.empty and any("speed" in i for i in issues)


def test_rejects_unknown_segment_and_bad_duration():
    clean, issues = validate(row(segment_id="X"), row(trip_id="t2", duration_s=-3))
    assert clean.empty and len(issues) >= 2


def test_rejects_real_name_as_collector():
    clean, issues = validate(row(collector_id="John Mensah"))
    assert clean.empty and any("pseudonymous" in i for i in issues)


def test_rejects_duplicate_trip_id_but_keeps_first():
    clean, issues = validate(row(), row())
    assert len(clean) == 1 and any("duplicate" in i for i in issues)


def test_missing_columns_reported():
    clean, issues = trips.validate_trips(pd.DataFrame({"trip_id": ["a"]}), SEG)
    assert clean.empty and "missing columns" in issues[0]


def test_append_trip_writes_header_once(tmp_path):
    path = tmp_path / "t.csv"
    trips.append_trip(path, row())
    trips.append_trip(path, row(trip_id="t2"))
    assert len(pd.read_csv(path)) == 2
