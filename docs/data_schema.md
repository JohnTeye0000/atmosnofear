# Preliminary Data Schema (draft — expect revision in Stage B)

All timestamps stored in **UTC** (Ghana is UTC+0 year-round, but store tz explicitly anyway). Every table carries a `source` / `data_label`.

## `segments` (static reference)

| Column | Type | Notes |
|---|---|---|
| segment_id | str | e.g. `KSI_ADUM_EJISU_NB` — stable key |
| name | str | human-readable |
| direction | str | inbound / outbound / compass |
| osm_way_ids | list[int] | geometry link |
| start_lat, start_lon, end_lat, end_lon | float | segment endpoints |
| length_m | float | |
| notes | str | stops, junctions, known chokepoints |

## `trips` (raw observations — real traffic data, **kept out of git**)

| Column | Type | Notes |
|---|---|---|
| trip_id | str | random, no link to a person |
| segment_id | str | FK → segments |
| start_time_utc | datetime | when the segment was entered |
| duration_s | float | travel time |
| vehicle_type | str | trotro / taxi / private / other |
| collector_id | str | **pseudonymous** participant code only |
| data_label | str | `observed` / `provider_estimate` / `synthetic` |
| quality_flag | str | ok / suspect / removed (+ reason) |

Constraint ideas for validation: `duration_s > 0`, plausible speed range (length_m / duration_s), no duplicate (segment, start_time, collector).

## `traffic_series` (derived, regular grid)

Aggregate trips to bins (start with 15 or 30 min).

| Column | Type | Notes |
|---|---|---|
| segment_id, bin_start_utc | key | |
| median_duration_s | float | median is robust to outliers |
| n_obs | int | **essential**: bins with n=0 are *missing*, not zero |
| is_imputed | bool | never silently fill |

## `weather_forecast` (features usable at prediction time)

| Column | Type | Notes |
|---|---|---|
| lat, lon | float | requested point (and grid point returned) |
| valid_time_utc | datetime | time the value is *for* |
| issued_time_utc | datetime | when forecast was made — **needed for leakage control; may not be exposed by the API (UNRESOLVED)** |
| temperature_2m, precipitation_mm, precipitation_probability, relative_humidity_2m, cloud_cover | float | check availability per model |
| model | str | e.g. Open-Meteo model name |
| data_label | `nwp_forecast` | |

## `weather_reanalysis` (oracle / evaluation only)

Same columns minus `issued_time_utc`, `data_label = reanalysis`. Never joined into deployable features.

## `weather_station` (optional, if GMet / gauge data obtained)

station_id, time_utc, rain_mm, temperature_c, data_label = `observed`.

## `examples` (model-ready, one row per prediction)

| Column | Notes |
|---|---|
| segment_id, t (prediction time), horizon_min | key |
| target_duration_s | duration observed at t+horizon |
| lag features | last observed duration(s) ≤ t, mean of last k bins, minutes since last observation |
| calendar | hour, day_of_week, is_weekend, Ghana public holiday flag |
| weather_fc_* | forecast features valid in window (t, t+h] from forecast issued ≤ t |
| weather_oracle_* | clearly named; for upper-bound experiment only |
| split | train / val / test — assigned chronologically with a gap |

## Why this shape
- Trips → bins keeps raw data private and gives a regular series.
- `n_obs` and `is_imputed` stop sparse crowd data from looking denser than it is.
- Separate forecast vs reanalysis tables make accidental leakage a *join the code refuses to do*, not a thing we must remember.
