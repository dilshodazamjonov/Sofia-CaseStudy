# Safia branch location model

Given a proposed latitude and longitude in Uzbekistan, the system builds pre-opening location features from
demographic, Safia-network, and OpenStreetMap data and predicts an expected Safia sales index.

**Start here:** [Quick start](#quick-start) · [Example output](#example-output) ·
[Final report](reports/final_report.md) · [Submission summary](reports/submission_summary.md)

## Overview

**Business problem.** Safia wants to estimate how well a proposed branch would sell before deciding to open
it. Before opening, the only thing known about a site is where it is. The model may therefore use only
information that exists before opening. Orders, average check, daily sales and anything else derived from a
branch's performance are never model inputs.

**Input.** The latitude and longitude of a proposed site in Uzbekistan.

**Output.**

| output | what it is |
|---|---|
| **Expected sales index** | the regression prediction: the expected average daily sales index of a branch at this site, on the supplied relative scale (see [Target](#target)) |
| **Relative location potential** (0–100) | a percentile: where that prediction ranks among the final model's out-of-fold predictions for the 127 existing branches |

**What the relative location potential means.**
- **Definition.** A potential of 23 means the site's predicted sales index is higher than about 23% of the
  out-of-fold predictions for the 127 existing branches. Each of those predictions was made by the same model
  setup while that branch's area was held out of training.
- **Not another sales prediction.** The potential ranks the site against the existing network and does not
  use actual sales.
- **Why it compares predictions.** Predictions are compressed: they span about 13–37, while actual values
  span 1.6–86.5. Against actual sales, every possible prediction would land between about the 30th and the
  91st percentile. Against other predictions, the score uses the full 0–100 range.

## Final approach

```text
lat/lon
→ administrative area        stat.uz district or city, by point-in-polygon on OpenStreetMap boundaries
→ demographics               6 district statistics from stat.uz
→ OSM commercial context     9 point-of-interest counts within 1 km
→ Safia-network context      7 distances to and counts of Safia's official stores
→ 22-feature vector
→ frozen ExtraTrees model    log1p on counts and distances; 500 trees; fitted once on all 127 branches
→ expected sales index       + relative location potential (percentile)
```

- **Same feature code for training and scoring.** The functions in `src/sofia_task/` that built the
  training rows also score a new site. For all 127 branches, the runtime feature vector reproduces the stored
  training row exactly, and the tests check this.
- **Model.** `ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=0)` on
  the raw target. It is saved with its preprocessing in `models/final_model.joblib`.
  `models/model_metadata.json` records the feature order, metrics and data snapshots.
- **Offline.** Prediction reads only local files and makes no network calls.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Run every command from the repository root.

```bash
uv sync
uv run pytest -q
```

- `uv sync` creates `.venv` with Python 3.13 (`.python-version`) and the exact package versions in
  `uv.lock`. It needs internet the first time; nothing after it does.
- `uv run pytest -q` should end with **`17 passed`** after about 5–10 seconds. The tests cover:
  - training/runtime feature parity for all 127 branches;
  - the feature schema;
  - prediction with the network blocked;
  - determinism and error handling;
  - the percentile layer and the web UI.

**Predict one location** (central Tashkent, Amir Temur square):

```bash
uv run python scripts/predict_location.py \
  --lat 41.3111 \
  --lon 69.2797
```

You should see three result lines (the output is shown under [Example output](#example-output)), followed
by the model's limitations. It takes about 2 seconds.

**Also print the 22 model inputs:**

```bash
uv run python scripts/predict_location.py \
  --lat 41.3111 \
  --lon 69.2797 \
  --features
```

The same result follows, then a `model features:` block with the 22 inputs in model order. For this site the
block shows, for example, 3 bakeries and 135 shops within 1 km, the nearest official Safia store at 0.47 km,
and 2 official stores within 1 km.

**Web UI:**

```bash
uv run streamlit run app.py
```

Open <http://localhost:8501>. Central Tashkent is pre-filled; press **Predict**. The page shows:
- the expected sales index and the relative location potential, with its percentile;
- the administrative area;
- the 22 model features, in a collapsed section;
- the limitations.

Stop the server with Ctrl+C.

**Options and errors.**
- **`--admin-code CODE`.** In four places OpenStreetMap does not separate a city from the district around it
  (Termez, Urgench, Khiva, Bayaut). There the CLI stops and lists the two stat.uz units to choose from, and
  the web UI shows a drop-down instead.
- **`--exclude-store-id NAME`.** Only for re-scoring an existing branch: it leaves out the branch's own entry
  in the official store list.
- **Unsupported locations.** A location outside Uzbekistan, or in a new district that stat.uz does not
  publish yet, gives a clear error instead of a guess.

**From Python:**

```python
from sofia_task.predict import predict_location
predict_location(41.3111, 69.2797)
# {'sales_index': 16.927..., 'location_potential': 22.834..., 'latitude': 41.3111, 'longitude': 69.2797,
#  'admin_code': '1726266', 'admin_name': 'Yunusabad district'}
```

## Example output

```text
$ uv run python scripts/predict_location.py --lat 41.3111 --lon 69.2797
Predicted Safia sales index: 16.9 / 100
Relative location potential: 23 / 100 (23rd percentile)
Administrative area: Yunusabad district (stat.uz 1726266)

Limitations:
- Predictions are compressed: out-of-sample predictions in cross-validation spanned about 13-37, ...
[four more limitation lines]
```

The web UI shows the same result as two headline numbers:

```text
Expected sales index            Relative location potential
16.9 / 100                      23 / 100
                                23rd percentile · Below average
```

- **Expected sales index (16.9)** is the regression prediction: the expected average daily sales index of a
  branch at this site, on the supplied scale (the median existing branch is at 17.8).
- **Relative location potential (23 / 100)** is a percentile. The prediction is higher than about 23% of
  the out-of-fold predictions for the 127 reference branches. It is **not** another sales prediction.
  - **Why a near-median index ranks low.** 16.9 is close to the median existing branch, yet the site ranks
    low. The model predicts more than 16.9 for about three quarters of the existing branches; their median
    prediction is 20.1.
  - **The label is descriptive.** "Below average" is one of five bands (Low, Below average, Typical, Above
    average, High). The bands are not business thresholds.
- **Administrative area** is the stat.uz district whose statistics were used. The number is its stat.uz code.
  The web UI shows only the name.

## Project structure

```text
pyproject.toml, uv.lock, .python-version   environment: Python 3.13 and the locked package versions
.streamlit/config.toml                     web UI settings: no usage statistics, no Deploy button
app.py               Streamlit web UI
src/sofia_task/      reusable code
  predict.py           lat/lon -> 22 features -> model -> sales index + potential (used by the CLI, UI and tests)
  demographics.py      point-in-polygon lookup of the stat.uz district
  osm_poi_features.py  POI category definitions and counts
  network_features.py  distances to and counts of Safia stores
  modeling.py          feature families, model pipelines, cross-validation loop
  accessibility_features.py   road features (investigated, not in the final model)
  download_stat_uz.py, build_stat_demography.py   stat.uz download and reference table
scripts/             data preparation, coordinate QA, experiments, final training, figures, prediction CLI
tests/               pytest suite for the predictor and the web UI
data/raw/            the supplied workbook
data/processed/      branch tables, coordinate corrections and QA, feature tables, modelling table, CV folds
data/external/       official Safia store list, stat.uz tables, OpenStreetMap-derived layers
models/              final_model.joblib and model_metadata.json
reports/             final report, submission summary, EDA report, result tables, out-of-fold predictions, figures
notebooks/           01_eda.ipynb, the target analysis behind reports/01_eda_report.md
```

The predictor reads only these files:
- `models/`;
- `reports/final_model_oof.csv`, the reference for the percentile;
- `data/external/osm/processed/osm_pois.csv`;
- `data/external/osm/uz_admin_areas.geojson`;
- `data/external/stat_uz/stat_demography.csv`;
- `data/processed/safia_official_stores_uz.csv`.

## Data sources

| # | source | content | files |
|---|---|---|---|
| 1 | Supplied Safia workbook | 127 branches (125 Safia, 2 Safia Mini) with location and daily sales and orders indices for 1–10 September 2026; iiko export generated 29.09.2026 | `data/raw/SampleSales_01-10_Sep_2026.xlsx`; tidy copies in `data/processed/branches.csv`, `daily_performance.csv`, `dataset_metadata.csv` |
| 2 | Safia official store list | safiabakery.uz/ru/locations, snapshot of 11 October 2026: 173 stores, 160 of them in Uzbekistan | `data/external/safia/safia_locations_2026-10-11.csv`, `data/processed/safia_official_stores_uz.csv` |
| 3 | Stat.uz | district and city demographic tables, 2026 values (growth against 2025) | `data/external/stat_uz/` |
| 4 | OpenStreetMap snapshot | Geofabrik extract of Uzbekistan, replication timestamp 2026-10-09T20:21Z: points of interest and district boundaries (roads were investigated) | `data/external/osm/` |

- **Snapshots, not live data.** Sources 2–4 were downloaded once and are stored as local snapshots.
  **Prediction does not use the internet**: one test blocks all network access and still predicts.
- **Workbook conversion.** The workbook was converted to the three CSVs once. There is no conversion script;
  `notebooks/01_eda.ipynb` checks the CSVs, for example that the supplied averages equal the recomputed
  daily means.
- **Large raw OSM files are not included.** They are the extract `data/external/osm/uzbekistan-latest.osm.pbf`
  (120 MB) and the road layer `data/external/osm/processed/osm_roads.gpkg` (164 MB), left out on purpose
  because of their size. Prediction, the tests and the experiments do not need them; see
  [Full research reproduction](#full-research-reproduction).

## Target

- **Definition.** The target is `avg_sales_index`: a branch's mean daily sales index over the 10 days from 1 to
  10 September 2026.
- **Daily index.** A branch's sales after discounts, divided by the highest single-day sales of any included
  branch, times 100. The maximum single-day value in the supplied data is therefore exactly 100 (Мега,
  Saturday 5 September).
- **Relative scale.** The index is not a currency amount and cannot be summed across branches as money.
  Branch averages range from 1.6 to 86.5 (median 17.8), strongly right-skewed.
- **What a prediction means.** Read it as early-September 2026 performance of an established branch.
- **Further analysis.** Target analysis is in [reports/01_eda_report.md](reports/01_eda_report.md). Daily
  rankings are stable (Spearman 0.90–0.97 between days), and the mean and the median of the daily values
  agree (r = 0.998).

## Data-quality work

- **Coordinate QA.** Every supplied branch coordinate was compared with Safia's official store list,
  matching on names and never fuzzily. Each correction was cross-checked against OpenStreetMap.
- **11 verified coordinate corrections.** Five points were copied from another branch (4.7–14.5 km off).
  Two were in the wrong city: W074 «Фергана» was supplied in Tashkent, and W174 «БУХ Вокзал» belongs at Kagan
  station. Four were at the wrong spot in the right city (0.9–7.4 km).
- **Correction layer kept separate.**
  - The supplied coordinates in `data/processed/branches.csv` are never edited.
  - Each correction, with its reason and source, is a row in `branch_coordinate_overrides.csv`.
  - `branch_coordinates.csv` combines the two.
  - `branch_coordinate_qa.csv` records the QA result for every branch, including six branches kept as
    supplied and flagged. W061 is the largest residual risk.
- **No post-opening leakage.**
  - Orders, average check, daily values, ranks and anything else derived from performance are never model
    inputs.
  - A branch's own official store is left out of its network features, so a training row describes the
    *other* stores around it, exactly as for a new site.
- **`universities_2km` removed.** Semantic QA showed that the OSM category behind it (`amenity=university`
  or `amenity=college`) mixed genuine higher-education institutions with other objects. Only 18% of the
  1,328 objects are named institutions. The rest are vocational colleges, lyceums and schools (52%),
  unnamed objects (20%), and faculties or campus buildings (6%). The feature was removed and not replaced.
- **District boundaries.** The common humanitarian boundary datasets (OCHA, geoBoundaries) are shifted
  about 4 km in Tashkent, so OpenStreetMap boundaries are used. Branches are matched to stat.uz units by
  point-in-polygon.

## Feature families

The final model uses 22 features:

| family | n | features |
|---|---|---|
| OSM points of interest within 1 km | 9 | bakeries and confectioneries, cafés, restaurants, supermarkets, all shops, schools, healthcare (hospitals, clinics, doctors), public-transport stops, all POIs |
| Official Safia network | 7 | distance to the nearest, 2nd and 3rd nearest official store; number of official stores within 0.5, 1, 2 and 3 km |
| Stat.uz demographics (district level) | 6 | population density, total population, population growth, urban share, working-age share (region level), share aged 20–29 |

Safia's own stores are never counted as POIs. Counts and distances get `log1p` inside the model pipeline,
and the feature files keep raw values.

**Investigated and rejected:**
- **Accessibility / roads.** Seven features: distance to a major road, road density, major-road length and
  intersections. They carried no signal on their own, and adding them made the models worse in spatial CV
  (ExtraTrees 7.99 → 8.15).
- **Noisy 500 m POI counts.** These had negative held-out permutation importance. Dropping them in favour
  of the 1 km counts improved spatial MAE from 7.99 to 7.59 and left random CV unchanged (8.70 over five
  seeds).
- **`universities_2km`.** The category was semantically wrong; see [Data-quality work](#data-quality-work).
- **Branch format.** Only two Safia Mini branches, too few to estimate a format effect.
- **Network measured against only the 127 labelled branches.** Results were equivalent. The official list
  is kept because a new site is scored against the full network.

## Validation

- **Unit of observation: the branch.** The 127 branches are the rows. The 1,270 branch-days are never
  separate rows, so a branch is never in both the training and the test part of a fold.
- **Random CV.** 5-fold, shuffled with seed 42, and the same repeated with five seeds because a single split
  of 127 branches is noisy.
- **Spatial CV (geography-aware).** The 104 branches within 25 km of central Tashkent are split into four
  k-means sectors, and the 23 branches farther out form a fifth group. Each group is held out once, so the
  model is always scored on an area it has not seen. Map:
  [reports/figures/cv_spatial_groups.png](reports/figures/cv_spatial_groups.png).
- **Spatial CV is the main decision metric** because the business task is predicting at new locations.
  The folds were fixed in `data/processed/cv_folds.csv` before any model was compared.
- **Metrics.** MAE is primary, in index points. RMSE and Spearman rank correlation are secondary. Each is
  given as mean ± standard deviation over the folds.

| validation | MAE | RMSE | Spearman | median-baseline MAE |
|---|---|---|---|---|
| **Spatial CV** (primary) | **7.59 ± 1.66** | 10.20 | 0.34 | 9.07 |
| Random CV, seed 42 | 8.31 ± 1.59 | 11.52 | 0.43 | 8.84 |
| Random CV, 5 seeds | 8.70 ± 1.57 | 12.03 | 0.32 | 8.94 |

**Spatial MAE improvement over the median baseline: 1.40 index points (95% bootstrap CI 0.40–2.33).** This
is pooled over the 127 branches' out-of-fold predictions. In random CV the improvement is 0.53 (CI −0.42 to
1.45), which includes zero. All experiment results are in
[reports/model_results.csv](reports/model_results.csv).

## Main findings

These are predictive associations across 127 existing branches, not causal claims.

- **Fewer nearby Safia branches are associated with stronger performance.**
  - Branches with no other official Safia store within 1 km have a median index of 20.9 (68 branches).
    Branches with one or two have 14.5 (51 branches).
  - This is the model's most important feature.
  - It fits cannibalisation, but it may also reflect where Safia has historically opened stores.
- **Stronger surrounding commercial activity is associated with stronger performance.** This means more
  bakeries and confectioneries, supermarkets and shops within 1 km, which is probably a proxy for footfall.
- **POI and Safia-network context contain most of the useful signal.**
  - Neither family alone beats the median baseline by much in spatial CV; combined, they do.
  - Together they carry almost all of the model's held-out permutation importance.
- **Demographics add little.** They are coarse, district-level values: 33 distinct districts across the 127
  branches.
- **Accessibility did not improve generalisation.** The road features were dropped.

## Limitations

- **Small sample.** Only 127 labelled branches.
- **Strong Tashkent concentration.** About 101 of the 127 branches are in Tashkent city, and regional towns
  are mostly represented by one branch each.
- **10-day target window.** Sales for 1–10 September 2026, with one weekend and one public holiday, so
  seasonality and trend cannot be assessed.
- **Uneven OpenStreetMap completeness.** Central Nukus is mapped far more densely than Tashkent, which
  inflates its POI counts. The Nukus branch is over-predicted by 26 points.
- **Compressed prediction range.** Out-of-fold predictions span about 13–37, while actual values span
  1.6–86.5.
- **Weak at identifying exceptional top sellers.** The strongest branches (Мега 86.5, Зенит 59.2, МедГор
  58.0) are predicted at 23–29. The model is better at separating weak, average and promising sites.
- **Safia Mini not modelled.** There are only two examples.
- **Residual data risks.**
  - One coordinate, W061 «Нукус», remains uncertain; only Safia can confirm it.
  - The public store list is incomplete: three labelled branches are missing from it, so the network
    features can miss a store.
- **Model-selection uncertainty.**
  - About 40 configurations were compared on the same folds, so the cross-validation estimates are
    slightly optimistic.
  - The spatial CV uses a single partition of the map, and its fold-to-fold spread is large (± 1.7 MAE).

## Full research reproduction

From the included feature tables, the whole modelling chain reruns in about 4 minutes:

```bash
uv run python scripts/build_modeling_table.py   # data/processed/modeling_table.csv
uv run python scripts/build_cv_folds.py         # data/processed/cv_folds.csv + reports/figures/cv_spatial_groups.png
uv run python scripts/run_experiments.py        # reports/model_results.csv, model_results_by_fold.csv
uv run python scripts/train_final_model.py      # models/, reports/feature_importance.csv, reports/final_model_oof.csv
uv run pytest -q
```

- **Reproducibility.** For this submission the rerun reproduced every output. The modelling table, CV
  folds, experiment results and out-of-fold predictions matched byte for byte, and the model file was
  identical.
- **Expected differences.** The metadata timestamp changes on every rerun. The permutation-importance table
  can differ in its last digit, because the forest adds up its trees in parallel.
- **Report figures.** `uv run python scripts/make_report_figures.py` redraws the three result figures in
  `reports/figures/` from the saved tables.

**Rebuilding the feature tables from the raw sources** is only partly possible from this submission. All
OpenStreetMap-derived data come from one Geofabrik snapshot, `uzbekistan-latest.osm.pbf` (replication
timestamp 2026-10-09T20:21Z). That snapshot is **not included** because of its size. Geofabrik serves only
the latest extract, so a new download is a different snapshot and will not reproduce the tables exactly.

| order | step | needs the original OSM snapshot? |
|---|---|---|
| 1 | `src/sofia_task/build_stat_demography.py` | no |
| 2 | `scripts/build_osm_districts.py` → `uz_admin_areas.geojson` | **yes**: downloads the current extract if the `.pbf` is missing |
| 3 | `scripts/build_branch_coordinates.py` | no |
| 4 | `scripts/build_branch_demographics.py` | no (reads the included `uz_admin_areas.geojson`) |
| 5 | `scripts/build_branch_network_features.py` | no |
| 6 | `scripts/qa_branch_coordinates.py` | no |
| 7 | `scripts/build_branch_public_network_features.py` | no |
| 8 | `scripts/extract_osm_pois.py` → `osm_pois.csv` | **yes** |
| 9 | `scripts/build_branch_osm_poi_features.py` | no (reads the included `osm_pois.csv`) |
| 10 | `scripts/extract_osm_roads.py` → `osm_roads.gpkg` | **yes** |
| 11 | `scripts/build_branch_accessibility_features.py` | needs `osm_roads.gpkg`, which is not included either (roads are not in the final model) |

- **Extraction steps.** Steps 2, 8 and 10 can only be reproduced exactly with the original `.pbf`.
- **Branch feature tables.** The other steps need only the included files. Run in the order above, they
  reproduced the included feature tables byte for byte, checked from a fresh copy for this submission.
- **Live downloaders.** `scripts/download_safia_locations.py` and `src/sofia_task/download_stat_uz.py` fetch
  live data and would create new snapshots. The snapshots used are included.
