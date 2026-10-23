# AtmosNoFear

Weather-aware travel-time prediction for road segments in Kumasi, Ghana.

**Research question:** On selected Kumasi road segments, does adding weather information available *at prediction time* improve travel-time prediction compared with otherwise identical models using traffic history alone?

> **Status: Stage B (data pipeline).** Real archived weather for Kumasi is fetched and cached. **No real traffic data exists yet and no model has been trained.** This repository makes no performance claims. A null result (weather does not help) is an acceptable outcome.

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

Traffic logging (once you have 1-3 real segments in `data/private/segments.csv`; see `configs/segments.example.csv` for the columns):

```bash
python -m atmosnofear.cli log-trip --segment SEG_ID --duration-s 840 --collector P001
python -m atmosnofear.cli validate
```

`data/private/` and `data/cache/` are git-ignored: trips are private, and weather is re-fetchable.

## Principles

- Real observations, reanalysis, forecasts, provider estimates and synthetic data are always labelled and never mixed.
- Chronological splits only; weather features are limited to what was available at prediction time.
- Simple baselines first; a neural network only if the data and results justify it.
- No raw location data or personal information in this repository. Trip collectors are pseudonymous codes (P001, ...), never names.

## Attribution

Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed CC BY 4.0. Road geometry (when added) © OpenStreetMap contributors, ODbL.
