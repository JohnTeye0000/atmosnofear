"""Command line entry points: python -m atmosnofear.cli <command>"""
from __future__ import annotations

import argparse
import tomllib
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from . import synthetic
from . import trips as trips_mod
from . import weather
from .binning import bin_trips
from .evaluate import evaluate_baselines


def _config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def cmd_update_weather(args) -> None:
    cfg = _config(args.config)
    loc, w, paths = cfg["location"], cfg["weather"], cfg["paths"]
    for kind in ("forecast", "reanalysis"):
        df = weather.update_cache(kind, loc["latitude"], loc["longitude"], w["variables"],
                                  date.fromisoformat(w["start_date"]), Path(paths["weather_cache_dir"]),
                                  model=w["forecast_model"] if kind == "forecast" else None)
        print(f"{kind}: {len(df)} hourly rows cached "
              f"({df['valid_time_utc'].min()} -> {df['valid_time_utc'].max()})")
    print("Weather data by Open-Meteo.com (CC BY 4.0)")


def cmd_log_trip(args) -> None:
    cfg = _config(args.config)
    row = {
        "trip_id": uuid.uuid4().hex[:12], "segment_id": args.segment,
        "start_time_utc": args.start or datetime.now(timezone.utc).isoformat(),
        "duration_s": args.duration_s, "vehicle_type": args.vehicle,
        "collector_id": args.collector, "data_label": "observed",
    }
    # Validate this single row before saving it.
    segments = pd.read_csv(cfg["paths"]["segments_file"])
    _, issues = trips_mod.validate_trips(pd.DataFrame([row]), segments)
    if issues:
        raise SystemExit("Trip rejected: " + "; ".join(issues))
    trips_mod.append_trip(Path(cfg["paths"]["trips_file"]), row)
    print(f"Logged trip {row['trip_id']}")


def cmd_validate(args) -> None:
    cfg = _config(args.config)
    trips = pd.read_csv(cfg["paths"]["trips_file"])
    segments = pd.read_csv(cfg["paths"]["segments_file"])
    clean, issues = trips_mod.validate_trips(trips, segments)
    print(f"{len(clean)}/{len(trips)} rows valid")
    for issue in issues:
        print(" -", issue)


def cmd_make_synthetic(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    synthetic.generate_trips(days=args.days, seed=args.seed).to_csv(out / "trips.csv", index=False)
    synthetic.synthetic_segments().to_csv(out / "segments.csv", index=False)
    print(f"Wrote SYNTHETIC test data to {out} (software testing only, not evidence about Kumasi)")


def cmd_evaluate_baselines(args) -> None:
    cfg = _config(args.config)
    trips_path = args.trips or cfg["paths"]["trips_file"]
    segments_path = args.segments or cfg["paths"]["segments_file"]
    clean, issues = trips_mod.validate_trips(pd.read_csv(trips_path), pd.read_csv(segments_path))
    for issue in issues:
        print("data issue:", issue)
    series = bin_trips(clean, bin_minutes=15)
    results = evaluate_baselines(series)
    label = results["data_label"].iloc[0]
    print(f"\nBaseline errors in minutes (data_label={label}; {len(clean)} trips, {len(series)} bins)")
    print(results.drop(columns="data_label").round(2).to_string(index=False))
    if label != "observed":
        print(f"\nWARNING: data is '{label}'. These numbers say nothing about real travel times.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"Saved {out}")


def main() -> None:
    p = argparse.ArgumentParser(prog="atmosnofear")
    p.add_argument("--config", default="configs/project.toml")
    sub = p.add_subparsers(required=True)
    sub.add_parser("update-weather", help="fetch only missing weather hours").set_defaults(fn=cmd_update_weather)
    lt = sub.add_parser("log-trip", help="record one consented trip")
    lt.add_argument("--segment", required=True)
    lt.add_argument("--duration-s", type=float, required=True)
    lt.add_argument("--vehicle", default="trotro")
    lt.add_argument("--collector", required=True, help="pseudonymous code, e.g. P001")
    lt.add_argument("--start", help="ISO time in UTC; default now")
    lt.set_defaults(fn=cmd_log_trip)
    sub.add_parser("validate", help="check the trip log").set_defaults(fn=cmd_validate)
    ms = sub.add_parser("make-synthetic", help="write labelled SYNTHETIC trips for testing")
    ms.add_argument("--out", default="data/synthetic")
    ms.add_argument("--days", type=int, default=28)
    ms.add_argument("--seed", type=int, default=0)
    ms.set_defaults(fn=cmd_make_synthetic)
    eb = sub.add_parser("evaluate-baselines", help="historical-mean and persistence baselines")
    eb.add_argument("--trips", help="override trips file from the config")
    eb.add_argument("--segments", help="override segments file from the config")
    eb.add_argument("--out", default="runs/baselines.csv")
    eb.set_defaults(fn=cmd_evaluate_baselines)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
