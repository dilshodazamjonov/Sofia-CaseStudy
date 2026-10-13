"""Freeze the two cross-validation views before any model is compared. The unit is one branch.

Input:   data/processed/modeling_table.csv
Outputs: data/processed/cv_folds.csv               (branch_code, standard_fold, spatial_group; table row order)
         reports/figures/cv_spatial_groups.png
Run: uv run python scripts/build_cv_folds.py

standard_fold: KFold(5, shuffle=True, random_state=42).
spatial_group: hold out whole areas (leave-one-group-out). K-means on all 127 points is useless here:
the far regional branches take the clusters and the whole Tashkent area stays one cluster of 105-113
branches until k = 15. So the 104 branches within 25 km of central Tashkent are split into 4 k-means
sectors (UTM 42N coordinates; k >= 5 spins off a 2-branch cluster), and the 23 branches farther out
(Tashkent-region towns and other regions) form a 5th group, "outside_tashkent". Uses coordinates only,
never the target.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "cv_folds.csv"
FIGURE = ROOT / "reports" / "figures" / "cv_spatial_groups.png"
CENTRE = (41.3111, 69.2797)  # Amir Temur square, Tashkent (lat, lon)
RADIUS_KM = 25
COMPASS = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"]

table = pd.read_csv(ROOT / "data" / "processed" / "modeling_table.csv")
folds = table[["branch_code"]].copy()
folds["standard_fold"] = 0
for fold, (_, test) in enumerate(KFold(5, shuffle=True, random_state=42).split(table)):
    folds.loc[test, "standard_fold"] = fold

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32642", always_xy=True)
x, y = to_utm.transform(table.longitude.to_numpy(), table.latitude.to_numpy())
cx, cy = to_utm.transform(CENTRE[1], CENTRE[0])
xy = np.c_[x - cx, y - cy] / 1000  # km from the centre
near = np.hypot(xy[:, 0], xy[:, 1]) <= RADIUS_KM
sector = KMeans(4, n_init=10, random_state=42).fit_predict(xy[near])
names = {}
for s in range(4):  # name each sector by the compass direction of its centroid
    mx, my = xy[near][sector == s].mean(axis=0)
    names[s] = "tashkent_" + COMPASS[int(round(np.degrees(np.arctan2(my, mx)) / 45)) % 8]
assert len(set(names.values())) == 4, f"two sectors share a name: {names}"
folds["spatial_group"] = "outside_tashkent"
folds.loc[near, "spatial_group"] = [names[s] for s in sector]
folds.to_csv(OUT, index=False)

# validation + report
check = pd.read_csv(OUT)
assert (check.branch_code == table.branch_code).all(), "row order differs from modeling_table.csv"
assert check.notna().all().all() and check.branch_code.is_unique
print(f"saved {OUT.relative_to(ROOT)}: {len(check)} rows")
print("standard_fold sizes:", check.standard_fold.value_counts().sort_index().to_dict())
summary = table.assign(group=check.spatial_group).groupby("group").agg(
    n=("branch_code", "size"), target_median=("avg_sales_index", "median"), target_max=("avg_sales_index", "max"),
    cities=("city_approx", lambda s: ", ".join(f"{c} {n}" for c, n in s.value_counts().items())))
print("\nspatial groups (each is held out once):")
print(summary.round(1).to_string())
print("\nbranches per standard_fold x spatial_group:")
print(pd.crosstab(check.spatial_group, check.standard_fold).to_string())

# map: whole country + Tashkent zoom; colour + marker shape + a direct label carry the group
groups = ["tashkent_northwest", "tashkent_northeast", "tashkent_south", "tashkent_southwest", "outside_tashkent"]
assert set(groups) == set(check.spatial_group)
colours = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#4a3aa7"]
markers = ["o", "s", "^", "D", "P"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})
for ax, zoom in zip(axes, [False, True]):
    for g, colour, marker in zip(groups, colours, markers):
        rows = table[check.spatial_group == g]
        ax.scatter(rows.longitude, rows.latitude, s=28, c=colour, marker=marker, edgecolors="white",
                   linewidths=0.8, label=f"{g} ({len(rows)})")
        if zoom and g != "outside_tashkent":
            ax.annotate(g.removeprefix("tashkent_"), (rows.longitude.mean(), rows.latitude.mean()),
                        fontsize=10, color="#0b0b0b", weight="bold", ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.set_aspect(1 / np.cos(np.radians(CENTRE[0])))
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.tick_params(colors="#52514e", labelsize=8)
    for side in ax.spines.values():
        side.set_color("#c3c2b7")
axes[1].set_xlim(69.05, 69.55)
axes[1].set_ylim(41.17, 41.47)
axes[0].set_title("All 127 branches", loc="left", fontsize=11, color="#0b0b0b")
axes[1].set_title("Tashkent area (104 branches within 25 km of the centre)", loc="left", fontsize=11, color="#0b0b0b")
axes[0].legend(loc="lower left", fontsize=8, frameon=False, title="spatial_group (n)", title_fontsize=8)
fig.suptitle("Spatial CV groups: each group is held out once", x=0.01, ha="left", fontsize=12, color="#0b0b0b")
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE, dpi=130, bbox_inches="tight", facecolor="white")
print(f"\nsaved {FIGURE.relative_to(ROOT)}")
