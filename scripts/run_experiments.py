"""Baselines, feature-family ablations, model comparison and robustness checks on the frozen CV folds.

Inputs:  data/processed/modeling_table.csv, data/processed/cv_folds.csv (scripts/build_cv_folds.py)
         data/processed/daily_performance.csv (only for the median-target robustness check)
Outputs: reports/model_results.csv          (one row per experiment x feature set x model x target x validation)
         reports/model_results_by_fold.csv  (the same runs, one row per held-out fold)
Run: uv run python scripts/run_experiments.py   (about 2 minutes)

Every run uses the same two frozen fold columns. Metrics are computed per held-out fold (MAE primary,
RMSE, Spearman) and summarised as mean and std (ddof=1) over the folds. The one exception is the
"standard_5seeds" robustness view: the standard 5-fold CV repeated with shuffle seeds 0-4 (25 folds),
because a single random 5-fold split of 127 branches turned out to be noisy.

The last model-selection step is "final_check": the chosen ExtraTrees without the 500 m POI counts,
adopted because spatial MAE improved by 0.29 (7.93 -> 7.64, every held-out group better) while the
5-seed random CV stayed level (8.73 -> 8.71). Nothing was selected after it. Those numbers are from the
selection run of 2026-10-12; universities_2km was removed from FEATURE_GROUPS on 2026-10-13 and
everything was rerun without re-selecting, so this file's current results differ slightly.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

from sofia_task.modeling import FEATURE_GROUPS, TARGET, cross_validate, make_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "model_results.csv"
OUT_BY_FOLD = ROOT / "reports" / "model_results_by_fold.csv"

table = pd.read_csv(ROOT / "data" / "processed" / "modeling_table.csv")
folds = pd.read_csv(ROOT / "data" / "processed" / "cv_folds.csv")
assert (folds.branch_code == table.branch_code).all(), "cv_folds.csv is not in modeling_table.csv row order"
VALIDATIONS = {"standard": folds.standard_fold, "spatial": folds.spatial_group}
COORDS = ["latitude", "longitude"]

# robustness target: median instead of mean of the 10 daily sales indices
daily = pd.read_csv(ROOT / "data" / "processed" / "daily_performance.csv")
assert (daily.groupby("branch_code").size() == 10).all(), "a branch without exactly 10 days"
table["median_sales_index"] = table.branch_code.map(daily.groupby("branch_code").sales_index.median())


def families(*names):
    return [f for name in names for f in FEATURE_GROUPS[name]]


# feature set -> families; network_127 only ever replaces network_public, the two are never combined
PND = "pois+network+demographics"
ABLATION = {
    "demographics": ["demographics"],
    "pois": ["pois"],
    "network_public": ["network_public"],
    "network_127": ["network_127"],
    "accessibility": ["accessibility"],
    "pois+network": ["pois", "network_public"],
    "pois+demographics": ["pois", "demographics"],
    PND: ["pois", "network_public", "demographics"],  # = all - accessibility
    "pois+network127+demographics": ["pois", "network_127", "demographics"],
    "all": ["demographics", "pois", "network_public", "accessibility"],
    "all_network127": ["demographics", "pois", "network_127", "accessibility"],
    "all-demographics": ["pois", "network_public", "accessibility"],
    "all-pois": ["demographics", "network_public", "accessibility"],
    "all-network": ["demographics", "pois", "accessibility"],
}
MODELS = ["ridge", "ridge_logy", "extra_trees", "extra_trees_logy", "catboost", "catboost_logy"]  # _logy: fit log1p(y)
NO_TOTAL_POIS = [f for f in families(*ABLATION[PND]) if not f.startswith("total_pois")]
# final check (the last model-selection step): are 1 km POI counts enough? 500 m POI counts had negative
# held-out permutation importance. public_safia_count_500m is a network feature and stays.
POI_500M = [f for f in FEATURE_GROUPS["pois"] if f.endswith("_500m")]
NO_POI_500M = [f for f in families(*ABLATION[PND]) if f not in POI_500M]

runs = []  # (experiment, feature_set, model, features, target)
runs += [("baseline", "none", "mean", COORDS, TARGET), ("baseline", "none", "median", COORDS, TARGET)]  # X ignored
runs += [("baseline", "coordinates", "knn_coords", COORDS, TARGET)]
runs += [("ablation", name, model, families(*fams), TARGET) for model in ["ridge", "extra_trees"] for name, fams in ABLATION.items()]
# model comparison on the best ablation set (accessibility adds nothing) and on every main family
runs += [("models", name, model, families(*ABLATION[name]), TARGET) for name in [PND, "all"] for model in MODELS]
runs += [("robustness", PND + "-total_pois", "extra_trees", NO_TOTAL_POIS, TARGET)]
runs += [("robustness", "none", "median", COORDS, "median_sales_index"),
         ("robustness", PND, "extra_trees", families(*ABLATION[PND]), "median_sales_index")]
runs += [("final_check", PND + "-poi_500m", "extra_trees", NO_POI_500M, TARGET)]

rows, by_fold = [], []


def record(experiment, feature_set, model, features, target, validation, per_fold):
    key = {"experiment": experiment, "feature_set": feature_set, "model": model, "target": target,
           "validation_type": validation, "n_features": 0 if model in ("mean", "median") else len(features)}
    by_fold.extend({**key, **r} for r in per_fold.to_dict("records"))
    rows.append({**key, **{f"{m}_{s}": getattr(per_fold[m], s)() for m in ["MAE", "RMSE", "Spearman"] for s in ["mean", "std"]}})


for experiment, feature_set, model, features, target in runs:
    for validation, fold_col in VALIDATIONS.items():
        per_fold, _ = cross_validate(make_model(model, features), table[features], table[target], fold_col)
        record(experiment, feature_set, model, features, target, validation, per_fold)
    print(f"done: {experiment:<10} {feature_set:<34} {model:<16} {target}")

# robustness: the standard view repeated over 5 shuffle seeds, for the shortlisted models
SEED_RUNS = [("robustness", "none", "median", COORDS), ("robustness", "coordinates", "knn_coords", COORDS)]
SEED_RUNS += [("robustness", PND, model, families(*ABLATION[PND])) for model in MODELS]
SEED_RUNS += [("final_check", PND + "-poi_500m", "extra_trees", NO_POI_500M)]
for experiment, feature_set, model, features in SEED_RUNS:
    per_fold = []
    for seed in range(5):
        seed_folds = pd.Series(0, index=table.index)
        for fold, (_, test) in enumerate(KFold(5, shuffle=True, random_state=seed).split(table)):
            seed_folds[test] = fold
        result, _ = cross_validate(make_model(model, features), table[features], table[TARGET], seed_folds)
        per_fold.append(result.assign(fold=f"seed{seed}_" + result.fold.astype(str)))
    record(experiment, feature_set, model, features, TARGET, "standard_5seeds", pd.concat(per_fold, ignore_index=True))
    print(f"done: standard_5seeds {feature_set} {model}")

results = pd.DataFrame(rows)
results.round(4).to_csv(OUT, index=False)
pd.DataFrame(by_fold).round(4).to_csv(OUT_BY_FOLD, index=False)
print(f"\nsaved {OUT.relative_to(ROOT)} ({len(results)} rows) and {OUT_BY_FOLD.relative_to(ROOT)}")

# readable summary: one line per run, standard vs spatial side by side
pd.set_option("display.width", 250)
two_views = results[results.validation_type != "standard_5seeds"]
wide = two_views.pivot_table(index=["experiment", "feature_set", "model", "target", "n_features"], columns="validation_type",
                             values=["MAE_mean", "MAE_std", "RMSE_mean", "Spearman_mean"], sort=False)
fmt = "{:.2f}".format
print(pd.DataFrame({
    "std MAE": wide["MAE_mean"]["standard"].map(fmt) + " ± " + wide["MAE_std"]["standard"].map(fmt),
    "std RMSE": wide["RMSE_mean"]["standard"].round(2),
    "std rho": wide["Spearman_mean"]["standard"].round(2),
    "spatial MAE": wide["MAE_mean"]["spatial"].map(fmt) + " ± " + wide["MAE_std"]["spatial"].map(fmt),
    "spatial RMSE": wide["RMSE_mean"]["spatial"].round(2),
    "spatial rho": wide["Spearman_mean"]["spatial"].round(2),
}).to_string())
print("\nstandard CV repeated over 5 shuffle seeds (25 folds):")
print(results[results.validation_type == "standard_5seeds"][["feature_set", "model", "MAE_mean", "MAE_std", "RMSE_mean", "Spearman_mean"]]
      .round(2).to_string(index=False))

final_check = results[(results.model == "extra_trees") & (results.target == TARGET)
                      & results.feature_set.isin([PND, PND + "-poi_500m"])].drop_duplicates(["feature_set", "validation_type"])
print(f"\nfinal check: 500 m POI counts {POI_500M} removed (ExtraTrees, same folds and parameters):")
print(final_check.pivot_table(index="feature_set", columns="validation_type", values="MAE_mean", sort=False)
      [["spatial", "standard", "standard_5seeds"]].round(3).to_string())

fold_mae = pd.DataFrame(by_fold).query("validation_type == 'spatial'").pivot_table(
    index=["experiment", "feature_set", "model", "target"], columns="fold", values="MAE", sort=False)
print("\nspatial MAE per held-out group:")
print(fold_mae.round(2).to_string())
