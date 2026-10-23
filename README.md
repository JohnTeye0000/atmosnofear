# AtmosNoFear

Weather-aware travel-time prediction for road segments in Kumasi, Ghana.

**Research question:** On selected Kumasi road segments, does adding weather information available *at prediction time* improve travel-time prediction compared with otherwise identical models using traffic history alone?

> **Status: Stage B/C (pipeline and baselines).** Real archived weather for Kumasi is fetched and cached; baseline code is tested on synthetic data only. **No real traffic data exists yet and no results have been produced.** This repository makes no performance claims. A null result (weather does not help) is an acceptable outcome.

## Documents

- [docs/research_plan.md](docs/research_plan.md): research question, feasibility, literature, first experiment, criteria for a neural network
- [docs/data_sources.md](docs/data_sources.md): sources, licences, checklist, and measured findings
- [docs/data_schema.md](docs/data_schema.md): draft data schema

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
pytest
```

## Commands

Fetch only the weather hours not yet cached. It is safe to re-run at any time, so the dataset stays current:

```bash
python -m atmosnofear.cli update-weather
```

Define real segments from OpenStreetMap. The path is found on the road network (one-way roads respected) and measured along the road:

```bash
python -m atmosnofear.cli osm-roads                      # list named major roads
python -m atmosnofear.cli add-segment --id KSI_SUAME_AIRPORT_EB --name "Suame to Airport" \
    --direction eastbound --from "Suame Roundabout" --to "Airport Roundabout" --dry-run
```

`--from`/`--to` take an OSM name (e.g. a roundabout) or `lat,lon`. Drop `--dry-run` to save to `data/private/segments.csv` and `data/private/segment_geometry/`. Use `--roads "Accra Road,Osei Tutu II Boulevard"` to keep the path on the roads you actually travel. The path is the shortest drivable route on OSM, which may differ from a trotro's real route, so check the geometry file.

Traffic logging (needs segments in `data/private/segments.csv`; see `configs/segments.example.csv` for the columns):

```bash
python -m atmosnofear.cli log-trip --segment SEG_ID --duration-s 840 --collector P001
python -m atmosnofear.cli validate
```

Baselines (historical mean and persistence, chronological split, horizons 15/30/60 min). Try it now on labelled **synthetic** data, which tests the software and says nothing about Kumasi:

```bash
python -m atmosnofear.cli make-synthetic
python -m atmosnofear.cli evaluate-baselines --trips data/synthetic/trips.csv --segments data/synthetic/segments.csv
```

With real trips in `data/private/`, run `python -m atmosnofear.cli evaluate-baselines` and the output carries the data label.

## Code map

| Module | Purpose |
|---|---|
| `weather.py` | Open-Meteo client, incremental cache, forecast vs reanalysis kept apart |
| `segments.py` | OSM roads, directed path finding, segment definitions |
| `trips.py` | Trip log format and validation |
| `binning.py` | Trips to a regular 15-minute series (median, count) |
| `features.py` | Leakage-safe examples; weather attached only with matching data labels |
| `split.py` | Chronological train/val/test with a gap |
| `baselines.py` | Historical mean, persistence |
| `metrics.py` | MAE, RMSE, paired block bootstrap |
| `evaluate.py` | Baselines across horizons, stamped with the data label |
| `synthetic.py` | Labelled fake trips for software tests only |

`data/private/` and `data/cache/` are git-ignored: trips are private, and weather is re-fetchable.

## Principles

- Real observations, reanalysis, forecasts, provider estimates and synthetic data are always labelled and never mixed.
- Chronological splits only; weather features are limited to what was available at prediction time.
- Simple baselines first; a neural network only if the data and results justify it.
- No raw location data or personal information in this repository. Trip collectors are pseudonymous codes (P001, ...), never names.

## Attribution

Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed CC BY 4.0. Road geometry (when added) © OpenStreetMap contributors, ODbL.
