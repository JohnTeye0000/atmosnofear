"""Loader for MUTra-CDMX (Mexico City, Insurgentes Sur corridor).

Dataset: Rodriguez Roman, MUTra-CDMX v4.0 (release candidate), Zenodo,
DOI 10.5281/zenodo.21617011, CC BY 4.0 per the Zenodo page. The README inside the
archive says licence verification is still pending, and the traffic values are TomTom
output ("consult provider terms"), so the data stays out of git (data/external/).

Labels used here:
  - travel times  -> `provider_estimate` (TomTom commercial traffic service, not our observation)
  - dataset weather -> `provider_weather` (Meteosource hourly historical weather; observed-style,
    so it is NOT a forecast and is only comparable to an oracle upper bound)
Reconstructed rows (is_imputed) are dropped: they were filled partly by interpolation across
gaps, which uses future values and would contaminate a forecasting evaluation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .binning import SERIES_COLUMNS

LOCAL_TZ = "America/Mexico_City"
TRAFFIC_LABEL = "provider_estimate"
WEATHER_LABEL = "provider_weather"


def _read(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    local = df["datetime"].dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
    return df.assign(time_utc=local.dt.tz_convert("UTC")).dropna(subset=["time_utc"])


def load_series(csv_path: Path, bin_minutes: int = 15, include_imputed: bool = False) -> pd.DataFrame:
    """Median travel time per (segment, bin) in the standard series format."""
    df = _read(csv_path)
    if not include_imputed:
        df = df[~df["is_imputed"].astype(bool)]
    df = df.assign(bin_start_utc=df["time_utc"].dt.floor(f"{bin_minutes}min"), segment_id=df["segment_id"].astype(str))
    out = (df.groupby(["segment_id", "bin_start_utc"])
           .agg(median_duration_s=("current_travel_time", "median"), n_obs=("current_travel_time", "size"))
           .reset_index())
    return out.assign(data_label=TRAFFIC_LABEL)[SERIES_COLUMNS]


def load_provider_weather(csv_path: Path) -> pd.DataFrame:
    """One row per hour from the dataset's own weather columns (UTC)."""
    df = _read(csv_path)
    df = df.assign(valid_time_utc=df["time_utc"].dt.floor("h")).drop_duplicates("valid_time_utc")
    return pd.DataFrame({
        "valid_time_utc": df["valid_time_utc"], "precipitation": df["precipitation_hourly_total_mm"],
        "temperature_2m": df["temperature_c"], "wind_speed_ms": df["wind_speed_ms"],
        "data_label": WEATHER_LABEL}).sort_values("valid_time_utc").reset_index(drop=True)


def corridor_centre(segment_metadata_csv: Path) -> tuple[float, float]:
    """Mean of segment start points, used as the point for Open-Meteo weather requests."""
    m = pd.read_csv(segment_metadata_csv)
    return float(m["start_latitude"].mean()), float(m["start_longitude"].mean())
