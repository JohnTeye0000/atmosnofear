# AtmosNoFear

Weather-aware travel-time prediction for road segments in Kumasi, Ghana.

**Research question:** On selected Kumasi road segments, does adding weather information available *at prediction time* improve travel-time prediction compared with otherwise identical models using traffic history alone?

> **Status: Stage A — feasibility and planning.** No model has been trained, and no real traffic data is included. This repository makes no performance claims. A null result (weather does not help) is an acceptable outcome.

## Documents

- [docs/research_plan.md](docs/research_plan.md) — research question, feasibility, literature, first experiment, criteria for using a neural network
- [docs/data_sources.md](docs/data_sources.md) — data sources, licences, and checklist
- [docs/data_schema.md](docs/data_schema.md) — preliminary data schema

## Principles

- Real observations, reanalysis, forecasts, provider estimates and synthetic data are always labelled and never mixed.
- Chronological splits only; weather features are limited to what was available at prediction time.
- Simple baselines first; a neural network only if the data and results justify it.
- No raw location data or personal information in this repository.

## Attribution

Weather data (when added) by [Open-Meteo.com](https://open-meteo.com/), licensed CC BY 4.0. Road geometry (when added) © OpenStreetMap contributors, ODbL.
