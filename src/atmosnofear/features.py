"""Build model-ready examples with leakage guards.

Timing convention (read this before changing anything):
  - Bin b covers [b, b + bin). Its observations are only known at t = b + bin.
  - A prediction is made at time t. Horizon h predicts the bin that ENDS at t + h,
    i.e. the bin starting at t + h - bin.  (h=15, bin=15 -> the very next bin.)
  - Every traffic feature uses bins <= b only. Calendar features describe the
    target time, which is legitimately known in advance.
  - Weather features are added separately and are label-checked: forecast mode
    accepts only `nwp_forecast` data, oracle mode only `reanalysis`.
"""
from __future__ import annotations

import pandas as pd

MAX_STALE_BINS = 8          # a last observation older than this is treated as unknown
RAIN_THRESHOLD_MM = 0.1
EXAMPLE_COLUMNS = ["segment_id", "t_utc", "horizon_min", "target_bin_start", "target_duration_s",
                   "last_obs_s", "mean4_s", "stale_bins", "hour", "dow", "is_weekend", "tod_min"]
WEATHER_MODES = {"forecast": ("nwp_forecast", "wx_fc_"), "oracle": ("reanalysis", "wx_oracle_"),
                 "provider": ("provider_weather", "wx_prov_"), "placebo": ("placebo_weather", "wx_placebo_")}


def lag_columns(n_lags: int) -> list[str]:
    return [f"lag_{j}_s" for j in range(n_lags)]


def build_examples(series: pd.DataFrame, horizon_min: int, bin_minutes: int = 15, tz: str = "UTC",
                   n_lags: int = 0) -> pd.DataFrame:
    """One row per (segment, prediction time) for a single horizon.

    `tz` is the local time zone used for the calendar features (hour of day, weekday), so
    that rush hours line up with local time. Timestamps themselves stay in UTC.
    `n_lags` adds lag_0_s .. lag_{n-1}_s: the raw bin values at b, b-1, ... (NaN if the bin
    was empty). All lags are at or before the prediction time, so they are leakage-safe.
    """
    if horizon_min <= 0 or horizon_min % bin_minutes:
        raise ValueError(f"horizon_min ({horizon_min}) must be a positive multiple of bin_minutes ({bin_minutes})")
    k = horizon_min // bin_minutes
    bin_td = pd.Timedelta(minutes=bin_minutes)
    frames = []
    for seg, g in series.groupby("segment_id"):
        s = g.set_index("bin_start_utc")["median_duration_s"].sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq=f"{bin_minutes}min"))
        stale = s.groupby(s.notna().cumsum()).cumcount()  # bins since the last observation
        cols = {
            "segment_id": seg,
            "t_utc": s.index + bin_td,
            "horizon_min": horizon_min,
            "target_bin_start": s.index + k * bin_td,
            "target_duration_s": s.shift(-k).to_numpy(),
            "last_obs_s": s.ffill(limit=MAX_STALE_BINS).to_numpy(),
            "mean4_s": s.rolling(4, min_periods=1).mean().to_numpy(),
            "stale_bins": stale.to_numpy(),
        }
        cols.update({name: s.shift(j).to_numpy() for j, name in enumerate(lag_columns(n_lags))})
        frames.append(pd.DataFrame(cols))
    if not frames:
        return pd.DataFrame(columns=EXAMPLE_COLUMNS)
    ex = pd.concat(frames, ignore_index=True)
    ex = ex.dropna(subset=["target_duration_s", "last_obs_s"]).reset_index(drop=True)
    target = ex["target_bin_start"].dt.tz_convert(tz)
    return ex.assign(hour=target.dt.hour, dow=target.dt.dayofweek,
                     is_weekend=(target.dt.dayofweek >= 5).astype(int),
                     tod_min=target.dt.hour * 60 + target.dt.minute)


def add_weather(examples: pd.DataFrame, weather: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Attach weather valid at the target hour. Missing weather stays NaN (never filled)."""
    if mode not in WEATHER_MODES:
        raise ValueError(f"mode must be one of {sorted(WEATHER_MODES)}")
    label, prefix = WEATHER_MODES[mode]
    found = set(weather["data_label"].unique())
    if found != {label}:
        raise ValueError(f"{mode} weather must have data_label {label!r}, got {sorted(found)}")
    cols = [c for c in ("precipitation", "temperature_2m") if c in weather.columns]
    w = weather[["valid_time_utc", *cols]].drop_duplicates("valid_time_utc")
    w = w.rename(columns={c: prefix + c for c in cols})
    w = w.assign(valid_time_utc=w["valid_time_utc"].astype("datetime64[ns, UTC]"))
    hour = examples["target_bin_start"].dt.floor("h").astype("datetime64[ns, UTC]")
    out = (examples.assign(_hour=hour)
           .merge(w, left_on="_hour", right_on="valid_time_utc", how="left")
           .drop(columns=["_hour", "valid_time_utc"]))
    precip = out[prefix + "precipitation"]
    rain = (precip >= RAIN_THRESHOLD_MM).astype(float).where(precip.notna())
    return out.assign(**{prefix + "rain": rain})
