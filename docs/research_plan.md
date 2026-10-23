# Research Plan — AtmosNoFear

Weather-aware travel-time prediction for selected road segments in Kumasi, Ghana.

**Status: Stage A (feasibility). No model has been trained. No real traffic data exists in this repo yet. Nothing here is a result.**

Legend used throughout the docs:
- **[VERIFIED]** — checked against a source during Stage A (source linked).
- **[ASSUMPTION]** — reasonable but unchecked; must be confirmed.
- **[UNRESOLVED]** — an open question that blocks or shapes a decision.

---

## 1. Research question

**Primary (hypothesis, not conclusion):**
> On selected Kumasi road segments, does adding weather information available *at prediction time* reduce travel-time prediction error at 15-, 30- and 60-minute horizons, compared with otherwise identical models that use traffic history and calendar features alone?

- H0: weather features do not change out-of-sample MAE beyond noise.
- H1: weather features reduce MAE (most plausibly during/after rain and at peak hours).
- A null result is a legitimate, reportable outcome.

**Secondary (only if data allows):** Is the answer different for rain vs dry periods, peak vs off-peak, and by segment?

**Deliberately deferred:** building a "weather forecasting neural network". See §5 — forecasting rainfall at a point in Kumasi better than numerical weather prediction (NWP) is a much harder, separate problem, and the traffic question does not require it.

## 2. Critical feasibility assessment

| Component | Feasibility | Why |
|---|---|---|
| Weather inputs | **High** | Open-Meteo provides free historical reanalysis and archived forecasts (see `data_sources.md`). |
| Weather *ground truth* (observed rain) | **Low–Medium** | GMet station data is fee-based and sparse; Open-Meteo reanalysis is modelled, not observed. [UNRESOLVED] |
| Traffic travel-time data | **The blocker** | No known open historical travel-time dataset for Kumasi. Must be collected or obtained through a partner. |
| Neural network training | **Conditional** | Only meaningful with enough data (see §6). Do not start before the data gate is passed. |
| Dashboard | **Easy, but last** | A dashboard over no data is a demo, not research. |

**Main risks**
1. **No traffic data → no project.** Stage B cannot start until a collection route is chosen (§4).
2. **Too little rain in the data.** Kumasi has a bimodal rainy season (roughly Apr–Jul and Sep–Nov) [ASSUMPTION — confirm with climate normals]. A dry-season-only dataset *cannot* test the hypothesis, however large.
3. **Weather resolution mismatch.** Gridded weather (~2–25 km) vs. a few-hundred-metre road segment; tropical convective rain is very local. Weather may add little simply because it is too coarse. That would be a finding about data resolution, not about weather.
4. **Leakage.** Using observed/reanalysis weather for the prediction window gives the model information a real forecaster would not have (§7).
5. **Small, noisy, irregular samples.** Participant-collected trips are irregular in time, unlike sensor data. This shapes the entire data schema (see `data_schema.md`).
6. **Novelty overclaim.** A neural network for paratransit travel time in Kumasi already exists (§3).

## 3. Concise literature review

Honest scope: this is a *starting* review from targeted searches, not a systematic review. Every entry below must be read in full and its claims confirmed before it is cited in the final report. [UNRESOLVED]

### Kumasi / Ghana
- **CSIR-BRRI (2024), "An artificial neural network based model for predicting paratransit travel time"** — [BRRI publication page](https://brri.org/publications/2024-publications/refereed-journals/an-artificial-neural-network-based-model-for-predicting-paratransit-travel-time). As summarised from the listing: ANN trained on data collected onboard minibus taxis via a mobile app; a model with real-time information outperformed a history-only ANN; ANNs outperformed historical-average and regression baselines. 
  - **[UNRESOLVED]** The search did not confirm the study corridor (previously assumed to be Adum–Ejisu) or the authors' affiliations. Read the full paper before stating either.
  - **Implication:** the user cannot claim to be first to apply neural networks to Kumasi paratransit travel time. 
  - **Questions to answer from the paper:** Did it use weather? Sample size? Split method (chronological or random)? Horizon? Which baselines? Is the data available, or the authors contactable?

### Neural networks for travel time (general)
- Fu & Rilett (2000), ANN route travel-time estimation for dial-a-ride paratransit routing ([IDEAS/RePEc listing](https://ideas.repec.org/a/taf/transp/v24y2000i1p25-48.html)). Early evidence ANNs are viable for paratransit-type problems.
- ANN bus travel-time work (e.g. a Doha bus study reporting high R² on ~78k records, [Qatar Univ. record](https://journals.qu.edu.qa/index.php/CIC/article/view/3936)). Note: R² on a large, regular, sensor-rich bus dataset is not comparable to a small crowd-collected one.

### Weather and traffic prediction
- 2017 paper using deep belief networks and LSTM with rainfall as an input, reported better than without ([doi:10.1155/2017/6575947](https://www.doi.org/10.1155/2017/6575947)). **[UNRESOLVED]** Title and authors not confirmed in Stage A; open the DOI and record them before citing.
- LSTM/GRU travel-time forecasting from floating-car data with weather ([ASCE J. Transp. Eng. record](https://ascelibrary.com/doi/10.1061/JTEPBS.TEENG-7647)).
- Leeds 15/30/60-min traffic-flow MSc thesis using weather ([NCI repository](https://norma.ncirl.ie/2520/)) — a *thesis*, treat as weak evidence.
- Rainfall-intensity-aware speed prediction under extreme weather ([ITC article](https://itc.ktu.lt/index.php/ITC/article/view/43987/23607)).
- A thesis noting that rainfall effects do not scale smoothly with distance from stations and that weather data is sparser than traffic data ([DA-IICT repository](https://ir.daiict.ac.in/items/dbfacd41-e98d-442e-a101-559063d96b0c)).

### What the literature suggests (cautiously)
- Weather often helps in studies from temperate/high-sensor settings; effect size varies and baselines are sometimes weak.
- Much published work has uncertain leakage hygiene (observed weather used as a feature). **A rigorous forecast-only, chronological evaluation is a genuine, defensible differentiator.**

### Gap and defensible contribution
Not "first neural network in Kumasi." Instead, a **carefully controlled test** of one question:
1. Matched models with/without weather, identical everything else.
2. Weather features restricted to what was *knowable at prediction time* (archived forecasts), plus a clearly-labelled "oracle weather" upper bound.
3. Chronological split, multiple horizons, paired uncertainty on the MAE difference.
4. A documented, ethical, reusable **data-collection method** for a data-poor setting.

## 4. Traffic-data acquisition options (the decision that gates everything)

| Option | Pros | Cons / checks |
|---|---|---|
| **A. Consented participant GPS/trip logging** (own app or an existing GPS logger) | Real, local, permission-clean; matches the prior Kumasi method | Irregular sampling; small N; needs consent form, anonymisation, possibly university ethics review; slow |
| **B. Partner with CSIR-BRRI / KNUST** | May provide existing data, expertise, credibility | Dependent on goodwill; data-sharing agreement needed |
| **C. Commercial traffic-aware routing APIs polled on schedule** (Google Routes, TomTom, HERE…) | Regular sampling, easy to automate | **Caching/storage terms may forbid building a stored historical dataset** [UNRESOLVED — read ToS; Google's Routes policy page says caching of most content is restricted]; it is a *model estimate*, not observed travel; cost |
| **D. Open datasets** | Free | No Kumasi travel-time dataset identified [UNRESOLVED — search further: Mapillary, OSM GPS traces are not travel times] |

**Recommendation:** pursue **B and A in parallel**; treat C as a possible *supplement* only after confirming terms in writing, and label it "provider-estimated", never "observed". Never mix estimated and observed travel times in one column without a `source` field.

## 5. Weather: forecasting vs. correcting vs. just using

The original brief proposes forecasting temperature/rain. Recommended re-scoping:

1. **Stage C–E use archived NWP forecasts as features.** No weather model of our own.
2. **Optional later (Stage D+):** a small *local forecast-correction* model, evaluated against the NWP forecast itself, and only if a trustworthy rain ground truth exists (GMet station or a rain gauge). Without ground truth we can only compare forecast to reanalysis (model vs model) — which is not evidence of real accuracy.
3. Temperature is easy and low-value for traffic; **rain occurrence/intensity is what matters** and is the hardest to forecast.

## 6. Minimum dataset for a credible baseline

All numbers are **[ASSUMPTION]** planning heuristics, to be revisited after seeing real variance.

- **Segments:** 2–3 (start with 1 to prove the pipeline). Each defined by OSM way IDs / start-end coordinates and a direction.
- **Time span:** ≥ 8–12 weeks continuous, **including at least part of a rainy season**. Longer is better; seasonal generalisation needs ≥ 1 year.
- **Density:** enough observations to form a 15-minute-bin series per segment during daytime (e.g. 06:00–20:00). Irregular crowd data may need coarser bins (30-min) or aggregation.
- **Rain coverage:** the *test* period must contain a meaningful number of rainy bins (target: dozens of distinct rain events; at minimum a few hundred rainy-bin observations). If not, the with/without-weather comparison has no power and we say so.
- **Quality gate:** report % missing, % imputed, sample counts per segment × hour × weather condition before any modelling.
- **Weather:** Open-Meteo archived forecasts + reanalysis for the same period (cheap; no gate).

**Go / no-go gate for Stage C:** real travel-time observations for ≥1 segment spanning ≥ 8 weeks with documented provenance and consent. Until then we build and test only the weather client and pipeline plumbing against **synthetic data labelled `source=synthetic`**.

## 7. Leakage rules (non-negotiable)

For a prediction made at time *t* for time *t+h*, a feature is allowed only if it was available at *t*:
- Traffic: observations with timestamp ≤ *t*.
- Weather: a forecast *issued* at or before *t*, never observed/reanalysis weather for (t, t+h]. 
- **[UNRESOLVED]** Open-Meteo's Historical Forecast API is documented as built from successive model runs; confirm exactly which lead time each value reflects, so we know whether it is truly "what a forecaster had at *t*". Document the finding in `data_sources.md`.
- Two weather conditions are always labelled: `forecast_at_t` (deployable) and `oracle_observed` (upper bound, not deployable).
- Scalers/imputers are fit on train only. Chronological split with a gap ≥ max horizon between train/val/test.

## 8. Smallest defensible first experiment (Stage C)

On one to three segments, same train/val/test chronological split:

| # | Model | Purpose |
|---|---|---|
| B1 | Historical mean by (segment, day-of-week, time-bin) | How much is recurring pattern |
| B2 | Persistence (last observed travel time) | Does recency alone work |
| B3 | Regularised linear model, traffic+calendar only | Simple learned reference |
| M1 | Gradient boosting (`HistGradientBoostingRegressor`), traffic+calendar | Strong conventional model |
| M2 | Same as M1 **+ forecast weather features** | **The test of the hypothesis** |
| (M3) | Oracle-weather version of M2 | Upper bound on what weather could add |

- Metrics: MAE (headline, minutes), RMSE; per horizon, segment, hour, rain vs dry.
- Inference on **M1 vs M2**: paired difference in absolute errors, with a **block bootstrap** (blocks of days) confidence interval — consecutive observations are not independent.
- Fixed seeds, config files, all splits saved.

## 9. What would justify a neural network

Move to PyTorch (Stage D) only if **all** hold:
1. Real data passes the gate and has enough rows (rule of thumb [ASSUMPTION]: ≳ 10⁴ training examples; fewer → boosting/linear is more appropriate).
2. Baselines M1/M2 are implemented, tuned fairly, and reproducible.
3. There is a specific hypothesised structure a network could exploit (e.g. interactions across many segments, longer histories, joint multi-horizon output) that the boosted model demonstrably misses.

Order of complexity: small MLP (same features as M1/M2 → clean comparison) → only then sequence models (GRU/LSTM) if justified. A tuned MLP that fails to beat gradient boosting is a valid result to report.

## 10. Ethics and privacy

- Informed, written consent from participants; plain-language explanation of what is recorded and why.
- Collect minimum data: segment ID, direction, timestamp, duration. Do **not** store raw GPS traces, device IDs or names in the repo. Raw traces (if any) stay local and are coarsened/discarded after extracting segment durations.
- Check whether the university requires ethics review. [UNRESOLVED — ask supervisor/teacher]
- Never commit consent forms, participant lists or raw location data. `.gitignore` enforces `data/raw/` and `data/private/`.
- Respect each API's and dataset's terms (see `data_sources.md`).

## 11. Unresolved questions (for the user)

1. Do you have, or can you get, any real travel-time data right now? From where, and under what permission?
2. Which 1–3 segments in Kumasi do you travel regularly and could log yourself? (Your own consenting trips are a valid start.)
3. Is there a supervisor/teacher who can help with ethics sign-off and an introduction to KNUST/CSIR-BRRI?
4. What is the application deadline? (Determines whether you can cover a rainy season.)
5. Is a rain gauge / personal weather station feasible for ground truth?

## 12. Roadmap

| Stage | Output | Gate |
|---|---|---|
| **A** | These documents | Reviewed by user |
| **B** | Schema, Open-Meteo client, validators, synthetic-data generator (labelled) | Real data source chosen |
| **C** | EDA + baselines on real data | ≥ 8 weeks real data |
| **D** | PyTorch MLP | §9 criteria met |
| **E** | With/without-weather experiments, bootstrap CIs | — |
| **F** | Dashboard over *real* results | Stage E complete |
| **G** | Report, limitations, reproduction steps | — |
