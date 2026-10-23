# Results

**Scope.** Everything here is measured on one open dataset from another city: **MUTra-CDMX, Mexico City, Insurgentes Sur corridor, 1 Nov 2024 – 28 Feb 2025, 20 short segments (150 m – 2 km).** Travel times are TomTom's commercial estimates (`provider_estimate`), not our own observations. Findings describe this corridor and period only. **Nothing here is a finding about Kumasi or Ghana.** The data is not in this repository (see the README for how to obtain it).

Setup: 15-minute bins; horizons 15/30/60 minutes; chronological 60/20/20 split with a gap of at least the longest horizon; test period about 25 days of February (about 45,500 test examples per horizon). All errors are test-set MAE in **seconds**. Hyperparameters and early stopping use the validation fold only. Reproduce with `run-mutra` and `run-zoo` (README).

## 1. Short version

1. **Recent travel-time history helps, mostly at longer horizons.** Networks that see the last 3 hours of travel times and predict the change from the last value beat "same as the last trip" by about 6.5% at 15 minutes, 13% at 30 and 18% at 60 (0.7 s, 1.8 s and 3.0 s).
2. **There is no evidence that weather helps.** Real archived-forecast weather beat *shuffled* (meaningless) weather in only 6 of 24 model-horizon cells and did worse in 14.
3. **The statistical test was too lenient.** The shuffled placebo weather was flagged "significant" in 14 of 24 cells, so a single flagged result means little here.
4. **The data cannot test the weather question well anyway.** The period is Mexico City's dry season: only 2.8% of hours have measurable rain.

## 2. Traffic-only models

Test MAE in seconds (lower is better). "vs persistence" is the percentage reduction in error compared with predicting the last observed travel time.

| Model | 15 min | 30 min | 60 min |
|---|---|---|---|
| Historical mean (segment x weekday x time of day) | 28.10 | 28.11 | 28.13 |
| Persistence (last value) | 11.08 | 14.15 | 16.79 |
| Gradient boosting, level target | 11.21 | 13.60 | 15.53 |
| Gradient boosting, residual + 12 lags | 11.00 | 13.76 | 15.46 |
| MLP, level target | 11.00 | 13.45 | 15.25 |
| MLP, residual + 12 lags | 10.41 | 12.55 | 14.08 |
| GRU over 12 lags, residual | 10.40 | 12.32 | 13.83 |
| Quantile MLP (median), residual + lags | **10.36** | 12.34 | 14.11 |
| Quantile boosting (median), residual + lags | 11.00 | 13.73 | 15.46 |
| Blend of boosting, MLP and GRU | 10.51 | 12.64 | 14.10 |

Chosen **on the validation fold** (never on test): quantile MLP at 15 minutes (10.36 s, 6.5% better than persistence), GRU at 30 minutes (12.32 s, 13.0%) and at 60 minutes (13.83 s, 17.6%).

What to take from it, and what not to:
- Calendar averages alone are far worse than persistence, which suggests short-term persistence matters much more here than the typical daily pattern.
- At 15 minutes the models with no history beyond the last value only tie persistence (boosting is 1.2% worse).
- The gain comes from the history features and residual targets, mostly in the networks. Boosting with the same lags improves much less (7.9% at 60 minutes). That does **not** show networks beat boosting in general: the boosting grid was small (4 settings) and I tuned neither model extensively.
- The top networks (MLP with lags, GRU, quantile MLP, blend) are within about 0.3 s of each other at every horizon and used one training seed. **I cannot say the GRU is better than the MLP.** The blend did not beat its best member.

## 3. Does weather help?

Design: each model is fitted twice on identical splits and tuning, with and without weather columns. Four weather inputs were compared, all aligned to the target hour:
- `forecast`: Open-Meteo archived ECMWF forecast for the corridor (the only deployable input);
- `provider`: the dataset's own Meteosource weather (known after the fact, an upper bound);
- `placebo`: the forecast weather with whole local days shuffled between dates. It keeps the same values and day shapes but removes any true link to that day's traffic;
- a direct **forecast-versus-placebo** comparison, the key test.

Differences are the mean change in absolute error in seconds, with a block bootstrap by day (25 test days). Negative means the weather version had lower error.

| Comparison (24 cells = 8 models x 3 horizons) | lower error | no difference | higher error |
|---|---|---|---|
| real forecast vs traffic-only | 5 | 9 | 10 |
| placebo (shuffled) vs traffic-only | 8 | 10 | 6 |
| provider weather vs traffic-only | 1 | 16 | 7 |
| **real forecast vs placebo** | 6 | 4 | 14 |

For the three validation-selected models, real forecast weather was never better than the placebo:

| Selected model | forecast vs placebo (s) | 95% interval | verdict |
|---|---|---|---|
| quantile MLP, 15 min | +0.090 | 0.046 to 0.136 | forecast worse |
| GRU, 30 min | +0.235 | 0.089 to 0.384 | forecast worse |
| GRU, 60 min | +0.182 | -0.000 to 0.359 | no difference |

Interpretation:
- **The placebo showed the bootstrap is over-confident.** Meaningless shuffled weather was labelled a significant gain or loss in 14 of 24 cells (58%) when about 5% is the nominal rate. The test treats each fitted model as fixed and resamples only the test rows, so it misses the variation that comes from refitting a model with different input columns. Changing the inputs alone moved error by up to 0.19 s (mean 0.08 s) with random weather.
- **Real weather's effects sit in the same range as that noise.** The largest apparent gain was plain gradient boosting at 60 minutes, -0.50 s against traffic-only and -0.42 s against the placebo (about 3% of a 15.5 s error). It appeared in 1 of 3 boosting variants at that horizon; the others were about -0.09 s.
- **Where real weather did differ from the placebo, it was usually worse**, especially for the networks. One possible explanation (untested): training months (Nov-Jan) and the test month (Feb) have different weather, so real weather columns shift between train and test while shuffled ones do not.
- **The dataset cannot support a rain conclusion.** Only 81 of 2,880 hours (2.8%) have at least 0.1 mm of rain, 7 hours reach 1 mm, and the maximum is 2.8 mm per hour. The forecast flagged about 54 wet hours in the test period (4,320 flagged examples), more than the roughly 21 wet hours the dataset's own weather shows for all of February, so the forecast over-reports light rain here.
- With 96 weather comparisons, about 5 flagged cells are expected by chance alone even with a calibrated test.

**Conclusion: on this dataset, no evidence that weather information improves travel-time prediction, and no strong evidence against it either: the data has too little rain to tell.**

## 4. Prediction intervals

q10-q90 intervals (nominal coverage 0.80), traffic-only models, test period:

| Model | 15 min | 30 min | 60 min | mean width (s) |
|---|---|---|---|---|
| Quantile boosting | 0.859 | 0.854 | 0.849 | 28 - 38 |
| Quantile MLP | 0.813 | 0.868 | 0.811 | 28 - 37 |

Coverage is a little above nominal for boosting (0.85-0.86) and near it for the network, about 0.81-0.87. One weather variant (MLP, forecast weather, 30 min) fell to 0.702. Widths of 28-38 s are roughly 2.5 to 3 times the point-prediction error. Coverage is measured on a single February test period.

## 5. Limitations

- One corridor, one season (dry), four months, 20 short segments that share weather and traffic conditions, so they are not independent.
- Travel times are a commercial provider's estimates, not independent measurements. Persistence being hard to beat may partly reflect how those estimates are produced. Reconstructed rows were dropped, not used.
- One weather point for the whole corridor; one NWP model (ECMWF) chosen without comparison.
- One training seed for the zoo, a small boosting grid, no repeated splits. The first-run results below used 3 seeds.
- The historical-mean baseline is per segment, weekday and time of day (falling back to the segment mean for sparse cells). It ignores recent conditions, which is why it loses so heavily to persistence.
- The licence of the dataset is not fully resolved, so the data cannot be redistributed.
- Not done: per-segment and per-time-of-day error breakdowns, rainy-versus-dry error, a model-refit bootstrap, a second dataset with more rain.

## 6. First experiment (levels, 3 seeds, for the record)

The earlier `run-mutra` experiment (boosting and MLP with the level target, no lags) gave test MAE of 11.21 / 13.60 / 15.53 s (boosting) and 11.07 / 13.36 / 15.22 s (MLP) at 15 / 30 / 60 minutes, against persistence of 11.08 / 14.15 / 16.79 s. Its weather comparisons were inconsistent in sign between the two model types (boosting often slightly lower, MLP higher), which is what prompted the placebo test above. Oracle (reanalysis) weather did not behave like an upper bound either.

## 7. What would settle the weather question

1. Data covering a wet season, with at least several hundred hours of measurable rain in the test period.
2. A calibrated test: repeat the whole fit (not just resample the test rows) over several seeds and several placebo shuffles, and judge real weather against the placebo distribution.
3. Station or gauge rainfall as ground truth, so the forecast can be checked against reality.

## 8. Reproducibility

Code at the commit that added this file; configuration in `configs/mexico_city.toml`; seeds fixed (network seed 0, placebo seed 0). Weather comes from Open-Meteo (CC BY 4.0), the dataset from Zenodo DOI 10.5281/zenodo.21617011 (MD5 `b619f60eeca53b0b11191f7e979d2e28`). Cite the dataset and acknowledge TomTom and Meteosource.
