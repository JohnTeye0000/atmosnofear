# AtmosNoFear

Weather-aware travel-time prediction: gradient boosting and a PyTorch network, with and without weather.

**Research question:** Does adding weather information available *at prediction time* improve travel-time prediction compared with otherwise identical models using traffic history alone?

> **Study site.** The project began for Kumasi, Ghana, but no Kumasi travel-time data exists, so the reported experiments use an open dataset from another city: **MUTra-CDMX (Mexico City, Nov 2024 – Feb 2025)**. Findings describe that corridor only and say nothing about Ghana. The pipeline is location-agnostic, and the Kumasi tooling (OpenStreetMap segments, trip logging, weather cache) is kept for when real data exists. See `docs/research_plan.md` section 14 for the decision and its costs.
>
> A null result (weather does not help) is an acceptable outcome. Read `docs/RESULTS.md` for what was actually measured and the limits.

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

Gradient boosting and a PyTorch network, each with and without weather (needs `pip install -e ".[ml]"` and `update-weather` first):

```bash
python -m atmosnofear.cli compare-weather --trips data/synthetic/trips.csv --segments data/synthetic/segments.csv
python -m atmosnofear.cli compare-weather      # real trips from data/private/
```

Three feature sets per model: `traffic`, `traffic+forecast` (archived NWP forecast, deployable) and `traffic+oracle` (reanalysis, an upper bound only). The comparison is a paired block bootstrap by day. The output warns when there are too few rainy examples or test days, and that some "significant" cells are expected by chance across many comparisons. The synthetic run is a **negative control**: that traffic has no weather dependence, so weather should not help.

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
| `models.py` | Gradient boosting and a small PyTorch network on identical, train-fitted features |
| `experiment.py` | With vs without weather: same split, same tuning, paired bootstrap, power warnings |
| `deep.py` | GRU over travel-time history; quantile network; shared training loop |
| `zoo.py` | All architectures on one split, validation-based selection, placebo-weather test |
| `mutra.py` | Loader for the MUTra-CDMX proxy dataset (drops reconstructed rows, local time) |
| `evaluate.py` | Baselines across horizons, stamped with the data label |
| `synthetic.py` | Labelled fake trips for software tests only |

`data/private/` and `data/cache/` are git-ignored: trips are private, and weather is re-fetchable.

## Reproducing the Mexico City experiment

The dataset is not in this repository (licence status unresolved; travel times are TomTom output). Download `MUTra_CDMX_v4.0.rar` from [Zenodo, DOI 10.5281/zenodo.21617011](https://doi.org/10.5281/zenodo.21617011) (MD5 `b619f60eeca53b0b11191f7e979d2e28`), extract it so that `data/external/mutra/MUTra_CDMX_v4.0-rc1/data/MUTra_CDMX_completed.csv` exists, then:

```bash
python -m atmosnofear.cli --config configs/mexico_city.toml run-mutra
```

This fetches Open-Meteo weather for the corridor (cached), runs the matched with/without-weather experiment at 15/30/60 minutes, and writes `runs/mutra.csv` and `runs/mutra_comparisons.csv`. Cite the dataset and acknowledge TomTom and Meteosource.

The full model comparison (boosting, MLP, GRU over 12 lags, quantile models, blend, residual targets, validation-based selection, and a shuffled-weather placebo test) runs with:

```bash
python -m atmosnofear.cli --config configs/mexico_city.toml run-zoo --seeds 0 --epochs 30
```

It trains many models on a CPU and takes a long time (the runtime was not measured precisely; plan for well over half an hour). Results and caveats are in [docs/RESULTS.md](docs/RESULTS.md).

## Principles

- Real observations, reanalysis, forecasts, provider estimates and synthetic data are always labelled and never mixed.
- Chronological splits only; weather features are limited to what was available at prediction time.
- Simple baselines first; a neural network only if the data and results justify it.
- No raw location data or personal information in this repository. Trip collectors are pseudonymous codes (P001, ...), never names.

## Attribution

Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed CC BY 4.0. Road geometry (when added) © OpenStreetMap contributors, ODbL.
