# Safia branch location model

Safia wants to estimate how well a proposed new branch would sell, using only information that is
available before the branch opens. This repository contains the research behind that estimate and a
working offline predictor.

| input | output |
|---|---|
| latitude + longitude of a proposed site in Uzbekistan | **expected sales index** on the supplied 1–100 scale |
| | **relative location potential**: where that prediction ranks among the predictions for the existing branches (a percentile, 0–100) |

```text
$ uv run python scripts/predict_location.py --lat 41.3111 --lon 69.2797
Predicted Safia sales index: 16.9 / 100
Relative location potential: 23 / 100 (23rd percentile)
Administrative area: Yunusabad district (stat.uz 1726266)
```

The full write-up is in [reports/final_report.md](reports/final_report.md).

## Methodology

```text
127 labelled branches (supplied sales, 1–10 September 2026)
 → coordinate QA against Safia's official store list (11 corrections, auditable override layer)
 → demographics: stat.uz district statistics, matched by point-in-polygon
 → Safia network context: distances to and counts of official Safia stores
 → OSM commercial / location context: points of interest around the site
 → branch-level validation: random 5-fold CV and geography-aware (spatial) CV
 → feature-family ablations (Ridge and ExtraTrees)
 → model comparison (Ridge, ExtraTrees, CatBoost; raw and log target)
 → final ExtraTrees model (22 features)
```

Only information known before opening is used as a model input. Orders, average check, daily values
and anything derived from performance are excluded.

## Data sources

| # | source | what | files |
|---|---|---|---|
| 1 | Supplied case-study data | 127 branches with location, daily sales and orders indices for 1–10 Sep 2026 (iiko export generated 29.09.2026) | `data/raw/SampleSales_01-10_Sep_2026.xlsx`; tidy CSV copies in `data/processed/branches.csv`, `daily_performance.csv`, `dataset_metadata.csv` |
| 2 | Official Safia store list | safiabakery.uz/ru/locations, snapshot of 2026-10-11: 173 stores, 160 in Uzbekistan | `data/external/safia/`, `data/processed/safia_official_stores_uz.csv` |
| 3 | Stat.uz | district and city population tables, 2026 values (growth against 2025) | `data/external/stat_uz/` |
| 4 | OpenStreetMap | Geofabrik extract of Uzbekistan, replication timestamp 2026-10-09T20:21Z: points of interest, district boundaries, roads | `data/external/osm/` |

Sources 2–4 were downloaded once and are stored locally. **The predictor makes no live API or network
calls**; a test blocks all network access and predicts successfully. The supplied workbook was converted
to the three CSVs once; there is no conversion script, but `notebooks/01_eda.ipynb` checks the CSVs
(for example, the provided averages equal the recomputed daily means).

## Target

`avg_sales_index` is the mean of a branch's daily sales index over 1–10 September 2026. The daily index is
the branch's sales after discounts divided by the highest single-day sales of any included branch, times
100. The scale is relative: 100 is one branch-day (Мега, Saturday 5 September), and branch averages range
from 1.6 to 86.5 (median 17.8, strongly right-skewed). With ten days of data it should be read as
early-September 2026 performance. Target analysis: [reports/01_eda_report.md](reports/01_eda_report.md).

## Final features

22 features, computed at the site's coordinates:

| family | n | features |
|---|---|---|
| OSM points of interest within 1 km | 9 | bakeries and confectioneries, cafés, restaurants, supermarkets, all shops, schools, healthcare (hospitals, clinics, doctors), public-transport stops, all POIs |
| Official Safia network | 7 | distance to the nearest, 2nd and 3rd nearest official store; number of official stores within 0.5, 1, 2 and 3 km |
| Stat.uz demographics (district level) | 6 | population density, total population, population growth, urban share, working-age share (region level), share aged 20–29 |

Safia's own stores are never counted as POIs. Counts and distances get `log1p` inside the model pipeline;
the feature files keep raw values. Investigated and left out:

- **Road / accessibility features** (distance to a major road, road density, major-road length,
  intersections): no signal on their own, and adding them made every model worse in spatial CV.
- **POI counts within 500 m**: the 1 km counts generalised better to unseen areas (spatial MAE 7.99 → 7.59).
- **`universities_2km`**: removed after semantic QA showed it counted OSM objects tagged university *or*
  college, which are mostly vocational colleges, lyceums, unnamed objects and campus parts, not
  universities. Nothing replaced it.
- **Network against only the 127 labelled branches**: equivalent results; the official list is kept because
  a new site is scored against the full network.
- **Store format**: only two Safia Mini branches, too few to estimate a format effect.

## Validation

The unit of validation is one branch; the 1,270 branch-days are never used as separate rows. Two views:

- **Random CV**: 5-fold, shuffled (seed 42), and the same repeated with five seeds. It answers "how well
  does the model fit branches like the ones it has seen".
- **Spatial CV** (primary): the 104 branches within 25 km of central Tashkent are split into four k-means
  sectors (41, 24, 21 and 18 branches) and the 23 branches farther out form a fifth group; each group is
  held out in turn. This is the real task: predicting at locations the model has not seen. Map:
  [reports/figures/cv_spatial_groups.png](reports/figures/cv_spatial_groups.png).

Metrics: MAE (primary), RMSE and Spearman rank correlation, as mean ± standard deviation over folds.

| | MAE | RMSE | Spearman | median-baseline MAE |
|---|---|---|---|---|
| **Spatial CV** | **7.59 ± 1.66** | 10.20 | 0.34 | 9.07 |
| Random CV (seed 42) | 8.31 ± 1.59 | 11.52 | 0.43 | 8.84 |
| Random CV, 5 seeds | 8.70 ± 1.57 | 12.03 | 0.32 | 8.94 |

Over branches, the out-of-fold improvement on the median baseline is 1.40 MAE in spatial CV (95% bootstrap
CI 0.40–2.33) and 0.53 in random CV (CI −0.42 to 1.45). All experiment results:
[reports/model_results.csv](reports/model_results.csv).

## Main findings

These are predictive associations in 127 existing branches, not causal effects.

- **Nearby Safia saturation goes with weaker sales.** Branches with no other official Safia store within
  1 km have a median index of 20.9 (68 branches), against 14.5 for those with one or two (51); inside
  Tashkent alone the medians are 20.5 and 14.5. It is the model's most important feature.
- **Stronger surrounding commercial activity goes with stronger sales**: more bakeries and
  confectioneries, supermarkets and shops within 1 km.
- **POI and network context carry most of the usable signal, and only together.** On its own neither
  family beats the median baseline by much in spatial CV; combined they do.
- **Coarse demographics add relatively little.** They are district-level values (33 distinct districts
  across the 127 branches).
- **Road and accessibility features did not improve generalisation.**

## Limitations

- Only 127 labelled branches, and about 101 of them are in Tashkent city.
- The target covers 10 days of sales, including one weekend and one public holiday.
- OpenStreetMap completeness differs between cities; central Nukus is mapped far more densely than
  Tashkent, which inflates its POI counts (the Nukus branch is over-predicted by 26 points).
- Predictions are compressed: out-of-fold predictions span about 13–37 while actual values span
  1.6–86.5.
- The model is better at separating weak, average and promising sites than at identifying exceptional
  top sellers: the strongest branches (Мега 86.5, Зенит 59.2, МедГор 58.0) are predicted at 23–29.
- Safia Mini is not modelled: there are only two examples.
- The cross-validation estimates carry model-selection uncertainty: about 40 configurations were compared
  on the same folds.

## How to run

Requires [uv](https://docs.astral.sh/uv/). `uv sync` installs Python 3.13 (`.python-version`) and the packages pinned in `uv.lock`; this needs internet once, prediction itself does not.

```bash
uv sync                      # create the environment
uv run pytest -q             # 17 tests: feature reproduction, schema, offline prediction, determinism, UI

uv run python scripts/predict_location.py \
  --lat 41.3111 \
  --lon 69.2797              # add --features to also print the 22 model inputs

uv run streamlit run app.py  # minimal web UI on http://localhost:8501
```

CLI options: `--features` prints the model inputs; `--admin-code CODE` chooses the statistical unit when
the location is in a district that OSM does not split into district and city (Termez, Urgench, Khiva,
Bayaut; the error message lists the choices; the web UI shows a drop-down instead); `--exclude-store-id
NAME` is only for scoring an existing branch, leaving out its own entry in the official store list.
A location outside Uzbekistan, or in a new district that stat.uz does not publish yet, gives a clear error
instead of a guess.

From Python:

```python
from sofia_task.predict import predict_location
predict_location(41.3111, 69.2797)
# {'sales_index': 16.927..., 'location_potential': 22.834..., 'latitude': 41.3111, 'longitude': 69.2797,
#  'admin_code': '1726266', 'admin_name': 'Yunusabad district'}
```

The relative location potential is the percentile of the prediction among the final model's out-of-fold
predictions for the 127 reference branches (`reports/final_model_oof.csv`). It is not a second sales
forecast.

### Re-running the research

From the frozen feature tables (about 4 minutes):

```bash
uv run python scripts/build_modeling_table.py   # data/processed/modeling_table.csv
uv run python scripts/build_cv_folds.py         # data/processed/cv_folds.csv + spatial group map
uv run python scripts/run_experiments.py        # reports/model_results.csv (baselines, ablations, models, robustness)
uv run python scripts/train_final_model.py      # models/, reports/feature_importance.csv, reports/final_model_oof.csv
uv run python scripts/make_report_figures.py    # reports/figures/
uv run pytest -q
```

Rebuilding the feature tables from the raw sources, in order: `src/sofia_task/build_stat_demography.py`,
`scripts/build_osm_districts.py`, `build_branch_coordinates.py`, `build_branch_demographics.py`,
`build_branch_network_features.py`, `build_branch_public_network_features.py`, `qa_branch_coordinates.py`,
`extract_osm_pois.py`, `build_branch_osm_poi_features.py`, `extract_osm_roads.py`,
`build_branch_accessibility_features.py`. This needs the cached OSM extract
(`data/external/osm/uzbekistan-latest.osm.pbf`, 120 MB). If the file is missing,
`build_osm_districts.py` downloads the current extract from Geofabrik, which is a different snapshot and
will not reproduce the tables exactly.

## Repository layout

```text
app.py                      Streamlit UI
src/sofia_task/             reusable code: feature functions, modelling, predictor
  predict.py                  lat/lon -> features -> model -> sales index + potential
  demographics.py             point-in-polygon district lookup
  osm_poi_features.py         POI category definitions and counts
  network_features.py         official-network distances and counts
  accessibility_features.py   road features (investigated, not in the final model)
  modeling.py                 feature groups, model pipelines, cross-validation loop
scripts/                    data preparation, experiments, final training, CLI, figures
tests/                      pytest suite for the predictor and the UI
data/raw/                   supplied workbook
data/external/              official store list, stat.uz tables, OSM extract and derived layers
data/processed/             branch tables, coordinate QA, feature tables, modelling table, CV folds
models/                     final_model.joblib + model_metadata.json (features, metrics, data snapshots)
reports/                    EDA report, final report, results tables, figures
notebooks/01_eda.ipynb      target analysis behind reports/01_eda_report.md
```

Files needed for prediction: `models/`, `reports/final_model_oof.csv`,
`data/external/osm/processed/osm_pois.csv`, `data/external/osm/uz_admin_areas.geojson`,
`data/external/stat_uz/stat_demography.csv` and `data/processed/safia_official_stores_uz.csv`.
The two largest files, `data/external/osm/uzbekistan-latest.osm.pbf` (120 MB) and
`data/external/osm/processed/osm_roads.gpkg` (164 MB), are only needed to rebuild features from raw OSM.
They are listed in `.gitignore`, so a git checkout will not contain them. Prediction, the tests and the
experiments do not need them. Keep the local copies if a full rebuild matters: a fresh Geofabrik
download is a different snapshot.
