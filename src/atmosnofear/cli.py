"""Command line entry points: python -m atmosnofear.cli <command>"""
from __future__ import annotations

import argparse
import tomllib
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from . import segments as seg_mod
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


def _osm(args, cfg) -> list[dict]:
    cache = cfg["paths"].get("osm_cache", "data/cache/osm_kumasi_roads.json")
    return seg_mod.fetch_roads(Path(cache), refresh=getattr(args, "refresh", False))


def _point(text: str, elements: list[dict]) -> tuple[float, float]:
    """'lat,lon' or the name of an OSM way such as a roundabout."""
    parts = text.split(",")
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    return seg_mod.find_landmark(elements, text)


def cmd_osm_roads(args) -> None:
    summary = seg_mod.road_summary(_osm(args, _config(args.config)))
    print(summary.round({"km": 1}).to_string(index=False))
    print("\nRoad data (c) OpenStreetMap contributors (ODbL). Use road names or landmark names with add-segment.")


def cmd_add_segment(args) -> None:
    cfg = _config(args.config)
    elements = _osm(args, cfg)
    roads = {r.strip() for r in args.roads.split(",")} if args.roads else None
    row, line = seg_mod.build_segment(elements, args.id, args.name, args.direction,
                                      _point(args.start, elements), _point(args.end, elements), roads)
    print(f"{row['segment_id']}: {row['length_m'] / 1000:.2f} km "
          f"({row['start_lat']}, {row['start_lon']}) -> ({row['end_lat']}, {row['end_lon']}), {len(line)} points")
    print(row["notes"])
    if args.dry_run:
        print("dry run: nothing saved")
        return
    seg_mod.append_segment(Path(cfg["paths"]["segments_file"]), row)
    geo_dir = Path(cfg["paths"]["segments_file"]).parent / "segment_geometry"
    seg_mod.write_geojson(geo_dir / f"{row['segment_id']}.geojson", row["segment_id"], line)
    print(f"Saved to {cfg['paths']['segments_file']} and {geo_dir / (row['segment_id'] + '.geojson')}")


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
    orp = sub.add_parser("osm-roads", help="list named major roads in Kumasi from OpenStreetMap")
    orp.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    orp.set_defaults(fn=cmd_osm_roads)
    asg = sub.add_parser("add-segment", help="define a segment as a drivable path between two points")
    asg.add_argument("--id", required=True)
    asg.add_argument("--name", required=True)
    asg.add_argument("--direction", required=True, help="e.g. inbound, outbound, northbound")
    asg.add_argument("--from", dest="start", required=True, help="'lat,lon' or an OSM name (e.g. a roundabout)")
    asg.add_argument("--to", dest="end", required=True, help="'lat,lon' or an OSM name")
    asg.add_argument("--roads", help="comma-separated road names to restrict the path to")
    asg.add_argument("--refresh", action="store_true")
    asg.add_argument("--dry-run", action="store_true", help="show length and endpoints without saving")
    asg.set_defaults(fn=cmd_add_segment)
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
