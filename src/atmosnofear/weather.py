"""Open-Meteo client with an incremental local cache.

Two kinds of data are kept apart on purpose (see docs/research_plan.md section 7):
  - "forecast":   archived NWP forecasts  -> label `nwp_forecast` (usable as features)
  - "reanalysis": ERA5-style reanalysis   -> label `reanalysis`   (oracle/evaluation only)
Attribution: Weather data by Open-Meteo.com (CC BY 4.0).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ENDPOINTS = {
    "forecast": ("https://historical-forecast-api.open-meteo.com/v1/forecast", "nwp_forecast"),
    "reanalysis": ("https://archive-api.open-meteo.com/v1/archive", "reanalysis"),
}
TIMEOUT_S = 60


def parse_hourly(payload: dict, kind: str, retrieved_at: datetime, model: str | None = None) -> pd.DataFrame:
    """Turn an Open-Meteo JSON payload into a tidy DataFrame (times are UTC)."""
    if "hourly" not in payload:
        raise ValueError(f"Open-Meteo response has no 'hourly' block: {payload.get('reason', payload)}")
    url, label = ENDPOINTS[kind]
    df = pd.DataFrame(payload["hourly"])
    if model:  # with `models=` Open-Meteo may suffix columns, e.g. precipitation_ecmwf_ifs025
        df = df.rename(columns=lambda c: c.removesuffix(f"_{model}"))
    df["valid_time_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    # Open-Meteo snaps the request to a model grid point; keep where the data really is for.
    df["grid_lat"] = payload["latitude"]
    df["grid_lon"] = payload["longitude"]
    df["data_label"] = label
    df["model"] = model or "default_best_match"
    df["source_url"] = url
    df["retrieved_at_utc"] = pd.Timestamp(retrieved_at)
    return df


def _require_model(kind: str, model: str | None) -> None:
    # Measured in Stage B: without `models=`, the Historical Forecast API returned values
    # identical to the reanalysis archive (100% of hours), i.e. NOT a genuine forecast.
    if kind == "forecast" and not model:
        raise ValueError("forecast data needs an explicit NWP model (e.g. 'ecmwf_ifs025'); "
                         "the default series is identical to reanalysis")


def fetch(kind: str, lat: float, lon: float, start: date, end: date, variables: list[str],
          model: str | None = None) -> pd.DataFrame:
    """Download one date range. Raises on HTTP errors rather than failing silently."""
    _require_model(kind, model)
    url, _ = ENDPOINTS[kind]
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(variables), "timezone": "UTC",
    }
    if kind == "forecast":
        params["models"] = model
    resp = requests.get(url, params=params, timeout=TIMEOUT_S)
    resp.raise_for_status()
    return parse_hourly(resp.json(), kind, datetime.now(timezone.utc), model if kind == "forecast" else None)


def merge_cache(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Combine cached and fresh rows; fresh rows win on duplicate timestamps."""
    if old is None or old.empty:
        return new.sort_values("valid_time_utc").reset_index(drop=True)
    both = pd.concat([old, new], ignore_index=True)
    both = both.drop_duplicates("valid_time_utc", keep="last")
    return both.sort_values("valid_time_utc").reset_index(drop=True)


def next_start(cached: pd.DataFrame | None, default_start: date, overlap_days: int = 1) -> date:
    """Where to resume: a little before the last cached day, so late revisions are refreshed."""
    if cached is None or cached.empty:
        return default_start
    last_day = cached["valid_time_utc"].max().date()
    return max(default_start, last_day - timedelta(days=overlap_days))


def cache_path(cache_dir: Path, kind: str, lat: float, lon: float, model: str | None = None) -> Path:
    tag = f"{kind}_{model}" if kind == "forecast" else kind
    return Path(cache_dir) / f"weather_{tag}_{lat:.4f}_{lon:.4f}.csv"


def load_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["valid_time_utc", "retrieved_at_utc"])


def update_cache(kind: str, lat: float, lon: float, variables: list[str], default_start: date,
                 cache_dir: Path, end: date | None = None, model: str | None = None) -> pd.DataFrame:
    """Fetch only what is missing and save. Safe to re-run any time."""
    end = end or (date.today() - timedelta(days=1))  # yesterday: today is incomplete
    _require_model(kind, model)
    path = cache_path(cache_dir, kind, lat, lon, model)
    cached = load_cache(path)
    start = next_start(cached, default_start)
    if start > end:
        return cached if cached is not None else pd.DataFrame()
    merged = merge_cache(cached, fetch(kind, lat, lon, start, end, variables, model))
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged
