"""Trip log format and validation.

Privacy: a trip row holds only segment, start time, duration and a pseudonymous
collector code. No names, device IDs or raw GPS traces. The file lives under
data/private/ which is git-ignored.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

TRIP_COLUMNS = ["trip_id", "segment_id", "start_time_utc", "duration_s",
                "vehicle_type", "collector_id", "data_label"]
DATA_LABELS = {"observed", "provider_estimate", "synthetic"}
VEHICLE_TYPES = {"trotro", "taxi", "private", "other"}
COLLECTOR_RE = re.compile(r"^P\d{3,}$")  # e.g. P001, never a real name
MIN_KMH, MAX_KMH = 1.0, 120.0            # plausibility window for average speed


def append_trip(path: Path, row: dict) -> None:
    """Append one trip, creating the file with a header if needed."""
    missing = set(TRIP_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRIP_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({c: row[c] for c in TRIP_COLUMNS})


def validate_trips(trips: pd.DataFrame, segments: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (rows passing every check, issue messages). The input is not modified."""
    issues: list[str] = []
    absent = [c for c in TRIP_COLUMNS if c not in trips.columns]
    if absent:
        return trips.iloc[0:0], [f"missing columns: {absent}"]

    df = trips.copy()
    df["start_time_utc"] = pd.to_datetime(df["start_time_utc"], utc=True, errors="coerce")
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    length = df["segment_id"].map(segments.set_index("segment_id")["length_m"])
    kmh = (length / df["duration_s"]) * 3.6

    checks = {
        "unparseable start_time_utc": df["start_time_utc"].isna(),
        "non-positive or non-numeric duration_s": ~(df["duration_s"] > 0),
        "unknown segment_id": length.isna(),
        "implausible average speed": length.notna() & ~kmh.between(MIN_KMH, MAX_KMH),
        "invalid data_label": ~df["data_label"].isin(DATA_LABELS),
        "invalid vehicle_type": ~df["vehicle_type"].isin(VEHICLE_TYPES),
        "collector_id not pseudonymous (expected P001 style)":
            ~df["collector_id"].astype(str).str.match(COLLECTOR_RE),
        "duplicate trip_id": df["trip_id"].duplicated(keep="first"),
    }
    bad = pd.Series(False, index=df.index)
    for message, mask in checks.items():
        if mask.any():
            issues.append(f"{int(mask.sum())} row(s): {message}")
        bad |= mask
    return df[~bad].reset_index(drop=True), issues
