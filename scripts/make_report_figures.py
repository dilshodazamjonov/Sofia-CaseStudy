"""Three figures for reports/final_report.md, drawn from saved artifacts only (no model is fitted here).

Inputs:  reports/model_results.csv, reports/final_model_oof.csv, reports/feature_importance.csv
Outputs: reports/figures/model_vs_baseline_mae.png, oof_predicted_vs_actual.png, feature_importance.png
Run: uv run python scripts/make_report_figures.py   (after scripts/train_final_model.py)
The fourth report figure, cv_spatial_groups.png, comes from scripts/build_cv_folds.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e4e3df", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titlelocation": "left"})


def style(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {(FIGURES / name).relative_to(ROOT)}")


# 1. cross-validated MAE: final model vs baselines and the compared models (spatial CV and 5-seed random CV)
results = pd.read_csv(ROOT / "reports" / "model_results.csv")
results = results[results.target == "avg_sales_index"].set_index(["experiment", "feature_set", "model", "validation_type"]).MAE_mean
PND = "pois+network+demographics"
rows = [  # label, (spatial key), (5-seed random CV key); bottom to top
    ("Median baseline", ("baseline", "none", "median"), ("robustness", "none", "median")),
    ("Coordinates only (10 nearest branches)", ("baseline", "coordinates", "knn_coords"), ("robustness", "coordinates", "knn_coords")),
    ("Ridge, 29 features", ("models", PND, "ridge"), ("robustness", PND, "ridge")),
    ("CatBoost, 29 features", ("models", PND, "catboost"), ("robustness", PND, "catboost")),
    ("ExtraTrees, 29 features", ("models", PND, "extra_trees"), ("robustness", PND, "extra_trees")),
    ("Final: ExtraTrees, 22 features", ("final_check", PND + "-poi_500m", "extra_trees"), ("final_check", PND + "-poi_500m", "extra_trees")),
]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for y, (label, spatial_key, random_key) in enumerate(rows):
    spatial, random = results[spatial_key + ("spatial",)], results[random_key + ("standard_5seeds",)]
    ax.plot([spatial, random], [y, y], color=AXIS, linewidth=1.5, zorder=1)
    ax.scatter(spatial, y, s=55, color=BLUE, edgecolors="white", linewidths=2, zorder=2, label="Spatial CV (unseen areas)" if y == 0 else None)
    ax.scatter(random, y, s=55, color=ORANGE, edgecolors="white", linewidths=2, zorder=2, label="Random CV, 5 seeds" if y == 0 else None)
    if label.startswith(("Final", "Median")):  # label the two rows the comparison is about; above / below the dots
        ax.annotate(f"{spatial:.2f}", (spatial, y), xytext=(0, 7), textcoords="offset points", ha="center", color=INK, fontsize=8)
        ax.annotate(f"{random:.2f}", (random, y), xytext=(0, -13), textcoords="offset points", ha="center", color=INK, fontsize=8)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("MAE in sales-index points (lower is better)")
style(ax)
ax.grid(axis="y", visible=False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, frameon=False, fontsize=8)
ax.set_title("Cross-validated error: final model vs baselines")
save(fig, "model_vs_baseline_mae.png")

# 2. spatial-CV out-of-fold predictions vs actual
oof = pd.read_csv(ROOT / "reports" / "final_model_oof.csv")
fig, ax = plt.subplots(figsize=(5.6, 5.6))
ax.plot([0, 90], [0, 90], color=AXIS, linewidth=1, zorder=1)
ax.text(78, 84, "prediction = actual", color=INK2, fontsize=8, ha="right")
ax.scatter(oof.pred_spatial_group, oof.avg_sales_index, s=30, color=BLUE, edgecolors="white", linewidths=1.2, zorder=2)
oof["miss"] = (oof.pred_spatial_group - oof.avg_sales_index).abs()
for r in oof.nlargest(4, "miss").itertuples():  # the two near-identical points (Зенит, МедГор) get labels above / below
    dy = {"W025": 4, "W016": -10}.get(r.branch_code, -3)
    ax.annotate(f"{r.branch_code} {r.branch_name}", (r.pred_spatial_group, r.avg_sales_index), xytext=(6, dy),
                textcoords="offset points", color=INK2, fontsize=8)
ax.set_xlim(0, 90)
ax.set_ylim(0, 90)
ax.set_aspect("equal")
ax.set_xlabel("Out-of-fold prediction (spatial CV)")
ax.set_ylabel("Actual avg_sales_index")
ax.set_title("Predictions are compressed toward the middle")
ax.text(88, 3, f"predictions {oof.pred_spatial_group.min():.0f}-{oof.pred_spatial_group.max():.0f}, "
               f"actual {oof.avg_sales_index.min():.1f}-{oof.avg_sales_index.max():.1f}", color=INK2, fontsize=8, ha="right")
style(ax)
save(fig, "oof_predicted_vs_actual.png")

# 3. held-out permutation importance by feature, coloured by family, with the direction of the effect
importance = pd.read_csv(ROOT / "reports" / "feature_importance.csv").sort_values("perm_mae_increase")
colour = {"network_public": BLUE, "pois": ORANGE, "demographics": AQUA}
name = {"network_public": "Safia network", "pois": "OSM POIs", "demographics": "Demographics"}
fig, ax = plt.subplots(figsize=(7.5, 6.2))
y = range(len(importance))
ax.barh(y, importance.perm_mae_increase, height=0.6, color=importance.family.map(colour))
ax.errorbar(importance.perm_mae_increase, y, xerr=importance.perm_std_over_groups, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0)
right = (importance.perm_mae_increase + importance.perm_std_over_groups).max() + 0.08
for yi, r in zip(y, importance.itertuples()):
    ax.text(right, yi, f"{r.effect_p10_to_p90:+.1f}", va="center", ha="right", color=INK2, fontsize=8)
ax.text(right, len(importance) - 0.2, "p10 → p90\neffect", ha="right", va="bottom", color=INK, fontsize=8)
ax.axvline(0, color=AXIS, linewidth=1)
ax.set_yticks(list(y), importance.feature)
ax.set_xlabel("Increase in MAE when the feature is shuffled in a held-out spatial group")
ax.set_title("Feature importance of the final model")
ax.legend(handles=[Patch(color=colour[f], label=name[f]) for f in ["network_public", "pois", "demographics"]],
          loc="lower right", frameon=False, fontsize=8, bbox_to_anchor=(0.85, 0))
style(ax)
ax.grid(axis="y", visible=False)
save(fig, "feature_importance.png")
