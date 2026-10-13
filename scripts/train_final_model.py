"""Fit the selected model on all 127 branches, save it with its metadata, and explain it.

Inputs:  data/processed/modeling_table.csv, data/processed/cv_folds.csv
         reports/model_results.csv (scripts/run_experiments.py; the CV metrics stored in the metadata)
Outputs: models/final_model.joblib    (sklearn Pipeline: log1p on counts/distances -> ExtraTreesRegressor)
         models/model_metadata.json
         reports/feature_importance.csv  (permutation importance on held-out spatial groups + effect direction)
         reports/final_model_oof.csv     (out-of-fold predictions of the final model and the median baseline)
Run: uv run python scripts/train_final_model.py

Selection (reports/model_results.csv): ExtraTrees on POIs + official-network + demographics.
- spatial MAE 7.93: tied with CatBoost (7.76) and ExtraTrees on log1p(target) (7.89) within the fold
  noise (std ~1.7), clearly better than Ridge (8.57) and the median baseline (9.07);
- consistent on random CV (8.30 frozen split, 8.73 over 5 seeds), where CatBoost is not (8.97 / 9.04,
  no better than the median baseline at 8.84 / 8.94);
- raw target: the log-target variant ties on MAE but is worse on RMSE, so the simpler one wins;
- accessibility is left out: no signal on its own, and adding it worsened every model (ET 7.93 -> 8.13);
- the hyperparameters were set once by hand (min_samples_leaf 3-5 and max_features 0.33-0.5 all give
  7.8-7.9 spatial MAE);
- final check (experiment "final_check"): without the 7 POI counts at 500 m, spatial MAE 7.93 -> 7.64
  (every held-out group better) and 5-seed random CV 8.73 -> 8.71, so the 500 m POI counts are dropped.
Model selection stops here. The numbers above are from the selection run of 2026-10-12. On 2026-10-13
universities_2km was removed (an audit showed it counts college/lyceum objects, not universities) and the
whole chain was rerun with the same choices; the metadata reports the rerun's metrics.

Store format is not a feature: only 2 of the 127 branches are Safia Mini (see LIMITATIONS).
"""
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from sofia_task.modeling import FEATURE_GROUPS, LOG1P, TARGET, cross_validate, make_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "final_model.joblib"
METADATA_FILE = ROOT / "models" / "model_metadata.json"
IMPORTANCE_FILE = ROOT / "reports" / "feature_importance.csv"
OOF_FILE = ROOT / "reports" / "final_model_oof.csv"
FINAL_MODEL = "extra_trees"
FINAL_FAMILIES = ["pois", "network_public", "demographics"]
POI_500M = [f for f in FEATURE_GROUPS["pois"] if f.endswith("_500m")]  # dropped in the final check
FINAL_EXPERIMENT, FINAL_SET = "final_check", "pois+network+demographics-poi_500m"  # its rows in reports/model_results.csv
LIMITATIONS = [
    "The supplied data contain only two Safia Mini branches, both with relatively low sales indices. This was "
    "insufficient to estimate a robust format effect, so branch format was excluded from the final model.",
]

table = pd.read_csv(ROOT / "data" / "processed" / "modeling_table.csv")
folds = pd.read_csv(ROOT / "data" / "processed" / "cv_folds.csv")
assert (folds.branch_code == table.branch_code).all()
features = [f for family in FINAL_FAMILIES for f in FEATURE_GROUPS[family] if f not in POI_500M]
X, y = table[features], table[TARGET]

# 1. final model on every branch
model = make_model(FINAL_MODEL, features).fit(X, y)
MODEL_FILE.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_FILE)
print(f"saved {MODEL_FILE.relative_to(ROOT)}: {FINAL_MODEL} on {len(features)} features, {len(X)} branches")

# 2. out-of-fold predictions vs the median baseline; bootstrap CI (over branches) of the MAE gain
oof = table[["branch_code", "branch_name", "city_approx", TARGET]].copy()
for view in ["spatial_group", "standard_fold"]:
    oof[f"pred_{view}"] = cross_validate(model, X, y, folds[view])[1]
    oof[f"median_baseline_{view}"] = cross_validate(make_model("median", ["latitude"]), table[["latitude"]], y, folds[view])[1]
oof["spatial_group"] = folds.spatial_group
oof.round(3).to_csv(OOF_FILE, index=False)
rng = np.random.default_rng(0)
print("\nout-of-fold MAE gain over the median baseline (pooled over branches, 95% bootstrap CI):")
for view in ["spatial_group", "standard_fold"]:
    gain = (oof[f"median_baseline_{view}"] - y).abs() - (oof[f"pred_{view}"] - y).abs()
    boot = [rng.choice(gain, len(gain)).mean() for _ in range(2000)]
    print(f"  {view:<14} model MAE {np.mean(np.abs(oof[f'pred_{view}'] - y)):.2f}, baseline "
          f"{np.mean(np.abs(oof[f'median_baseline_{view}'] - y)):.2f}, gain {gain.mean():.2f} "
          f"[{np.percentile(boot, 2.5):.2f}, {np.percentile(boot, 97.5):.2f}]")

# 3. importance: MAE increase when one feature is shuffled in a held-out spatial group (model fitted
#    without that group), averaged over the 5 groups; direction from the full model: mean prediction
#    with the feature set to its 90th vs its 10th percentile for every branch (others unchanged)
per_group = []
for group in sorted(folds.spatial_group.unique()):
    test = (folds.spatial_group == group).to_numpy()
    fitted = make_model(FINAL_MODEL, features).fit(X[~test], y[~test])
    result = permutation_importance(fitted, X[test], y[test], scoring="neg_mean_absolute_error", n_repeats=20, random_state=0)
    per_group.append(pd.Series(result.importances_mean, index=features, name=group))
per_group = pd.concat(per_group, axis=1)
importance = pd.DataFrame({
    "feature": features,
    "family": [next(g for g in FINAL_FAMILIES if f in FEATURE_GROUPS[g]) for f in features],
    "perm_mae_increase": per_group.mean(axis=1).to_numpy(),
    "perm_std_over_groups": per_group.std(axis=1).to_numpy(),
    "p10": X.quantile(0.1).to_numpy(),
    "p90": X.quantile(0.9).to_numpy(),
})
effects = []
for f in features:
    low, high = X.copy(), X.copy()
    low[f], high[f] = X[f].quantile(0.1), X[f].quantile(0.9)
    effects.append(model.predict(high).mean() - model.predict(low).mean())
importance["effect_p10_to_p90"] = effects
importance = importance.sort_values("perm_mae_increase", ascending=False)
importance.round(4).to_csv(IMPORTANCE_FILE, index=False)
print(f"\nsaved {IMPORTANCE_FILE.relative_to(ROOT)}; features by held-out permutation importance:")
print(importance.round(3).to_string(index=False))
print("\nimportance summed by family:", importance.groupby("family").perm_mae_increase.sum().round(3).to_dict())

# 4. diagnostics on the spatial out-of-fold predictions
err = oof.pred_spatial_group - y
oof["abs_error"] = err.abs()
oof["error"] = err
print("\nlargest spatial out-of-fold misses:")
print(oof.nlargest(8, "abs_error")[["branch_code", "branch_name", "city_approx", TARGET, "pred_spatial_group", "error"]]
      .round(1).to_string(index=False))
in_tashkent = oof.city_approx == "Tashkent"
print(f"spatial OOF MAE: Tashkent {oof.abs_error[in_tashkent].mean():.2f} (n={in_tashkent.sum()}), "
      f"outside {oof.abs_error[~in_tashkent].mean():.2f} (n={(~in_tashkent).sum()})")
print(f"median absolute percentage error (spatial OOF): {np.median(oof.abs_error / y):.0%}")
print(f"spatial OOF predictions range {oof.pred_spatial_group.min():.1f}-{oof.pred_spatial_group.max():.1f} "
      f"(target {y.min():.1f}-{y.max():.1f})")

# 5. metadata
results = pd.read_csv(ROOT / "reports" / "model_results.csv")
metric_cols = ["MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "Spearman_mean", "Spearman_std"]


def cv_metrics(experiment, feature_set, model_name, validation):
    row = results[(results.experiment == experiment) & (results.feature_set == feature_set) & (results.model == model_name)
                  & (results.validation_type == validation) & (results.target == TARGET)]
    assert len(row) == 1, (experiment, feature_set, model_name, validation)
    return {c: None if pd.isna(row[c].iloc[0]) else round(float(row[c].iloc[0]), 3) for c in metric_cols}


params = model.named_steps["extratreesregressor"].get_params()
ablation_pnd = cv_metrics("ablation", "pois+network+demographics", FINAL_MODEL, "spatial")
metadata = {
    "model_file": str(MODEL_FILE.relative_to(ROOT)),
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "model_type": "sklearn Pipeline: ColumnTransformer(log1p on the listed columns, others passed through) -> ExtraTreesRegressor",
    "model_params": {k: params[k] for k in ["n_estimators", "min_samples_leaf", "max_features", "random_state"]},
    "target": {
        "name": TARGET,
        "definition": "mean of the daily sales index over 2026-09-01..2026-09-10 (10 days). Daily index = branch sales "
                      "after discounts / highest single-day sales of any included branch x 100 (iiko).",
        "training_range": [float(y.min()), float(y.max())],
        "training_median": float(y.median()),
        "interpretation": "early-September 2026 performance of an established branch; predictions are not clipped to 1-100",
    },
    "n_training_branches": int(len(X)),
    "features": features,
    "feature_families": {g: [f for f in FEATURE_GROUPS[g] if f in features] for g in FINAL_FAMILIES},
    "preprocessing": {
        "log1p": [f for f in features if f in LOG1P],
        "scaling": "none (tree model)",
        "missing_values": "none in training; every feature must be present at prediction time",
        "column_order": "pass a DataFrame with the columns in `features`",
    },
    "not_used": {
        "accessibility": "no signal on its own; adding it raised spatial MAE (ExtraTrees "
                         f"{ablation_pnd['MAE_mean']} -> {cv_metrics('ablation', 'all', FINAL_MODEL, 'spatial')['MAE_mean']})",
        "network_127": "same features against the 127 labelled branches only; robustness check, not combined",
        "poi_500m": {"features": POI_500M,
                     "reason": f"final check: removing them changed spatial MAE {ablation_pnd['MAE_mean']} -> "
                               f"{cv_metrics(FINAL_EXPERIMENT, FINAL_SET, FINAL_MODEL, 'spatial')['MAE_mean']} and 5-seed random CV "
                               f"{cv_metrics('robustness', 'pois+network+demographics', FINAL_MODEL, 'standard_5seeds')['MAE_mean']} -> "
                               f"{cv_metrics(FINAL_EXPERIMENT, FINAL_SET, FINAL_MODEL, 'standard_5seeds')['MAE_mean']} (selected on "
                               "2026-10-12 at 7.93 -> 7.64 and 8.73 -> 8.71, before universities_2km was removed)"},
        "universities_2km": "removed 2026-10-13: counts amenity=university|college OSM objects, which are mostly vocational "
                            "colleges, lyceums, unnamed objects and campus parts, not universities; not replaced",
        "branch_format": "see limitations",
        "leakage": ["avg_orders_index", "avg_check_index", "daily sales/orders", "ranks"],
    },
    "limitations": LIMITATIONS,
    "validation": {
        "unit": "one branch (the 1,270 branch-day rows are never split)",
        "spatial": {"design": "leave-one-group-out over 5 groups: 4 k-means sectors of the 104 branches within 25 km "
                              "of central Tashkent + all 23 branches farther out",
                    **cv_metrics(FINAL_EXPERIMENT, FINAL_SET, FINAL_MODEL, "spatial")},
        "standard": {"design": "KFold(5, shuffle=True, random_state=42)",
                     **cv_metrics(FINAL_EXPERIMENT, FINAL_SET, FINAL_MODEL, "standard")},
        "standard_5seeds": {"design": "KFold(5) repeated with shuffle seeds 0-4 (25 folds)",
                            **cv_metrics(FINAL_EXPERIMENT, FINAL_SET, FINAL_MODEL, "standard_5seeds")},
        "baseline_median_spatial": cv_metrics("baseline", "none", "median", "spatial"),
        "baseline_median_standard": cv_metrics("baseline", "none", "median", "standard"),
        "baseline_median_standard_5seeds": cv_metrics("robustness", "none", "median", "standard_5seeds"),
        "metrics_note": "MAE primary; mean and std (ddof=1) over folds. The final model was chosen on these same folds "
                        "from about 40 candidates, so the estimates are slightly optimistic.",
    },
    "data_snapshots": {
        "sales": "iiko export generated 2026-09-29, period 2026-09-01..2026-09-10 (data/processed/dataset_metadata.csv)",
        "branch_coordinates": "data/processed/branch_coordinates.csv, frozen 2026-10-11 (11 corrections via branch_coordinate_overrides.csv)",
        "safia_official_store_list": "safiabakery.uz/ru/locations retrieved 2026-10-11; 160 stores in Uzbekistan (safia_official_stores_uz.csv)",
        "osm": "Geofabrik uzbekistan-latest.osm.pbf, replication timestamp 2026-10-09T20:21:04Z",
        "stat_uz": "stat.uz district tables, 2026 values (growth vs 2025), downloaded 2026-10-11",
    },
    "scoring_a_new_site": f"compute the same {len(features)} features at the site's lat/lon: sofia_task.osm_poi_features.osm_poi_features, "
                          "sofia_task.network_features.public_network_features against all 160 official stores (nothing excluded), "
                          "and the stat.uz district of the site (as in scripts/build_branch_demographics.py)",
    "library_versions": {p: version(p) for p in ["scikit-learn", "numpy", "pandas", "joblib"]},
}
METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
print(f"\nsaved {METADATA_FILE.relative_to(ROOT)}")

# reload check: the saved model reproduces the in-memory predictions
assert np.allclose(joblib.load(MODEL_FILE).predict(X), model.predict(X))
print("reload check passed: saved model reproduces the training predictions")
