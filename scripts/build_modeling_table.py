"""One modeling row per branch: the target, every frozen location-feature family and a few metadata columns.

Inputs:  data/processed/branches.csv            (target avg_sales_index, branch_name, format, city_approx)
         data/processed/branch_coordinates.csv  (effective coordinates; the supplied ones are never used)
         data/processed/branch_demographics.csv, branch_osm_poi_features.csv,
         branch_public_network_features.csv, branch_network_features.csv, branch_accessibility_features.csv
Output:  data/processed/modeling_table.csv
Run: uv run python scripts/build_modeling_table.py

Model inputs are exactly the columns in sofia_task.modeling.FEATURE_GROUPS; META and the target are kept
for diagnostics only. Left out on purpose: avg_orders_index / avg_check_index (performance = leakage),
admin_code (an identifier), road_length_*_km (= road_density_* x the constant buffer area).
"""
from pathlib import Path

import pandas as pd

from sofia_task.modeling import FEATURE_GROUPS, META, TARGET

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "modeling_table.csv"
FEATURE_FILES = ["branch_demographics.csv", "branch_osm_poi_features.csv", "branch_public_network_features.csv",
                 "branch_network_features.csv", "branch_accessibility_features.csv"]
LEAKY = {"avg_orders_index", "avg_check_index", "sales_index", "orders_index", "date", "rank"}
# city_approx was derived from the supplied coordinates; W074's corrected point is in Fergana, 235 km from Tashkent
CITY_FIX = {"W074": "Fergana"}

table = pd.read_csv(PROCESSED / "branches.csv")[["branch_code", "branch_name", "format", "city_approx", TARGET]]
coords = pd.read_csv(PROCESSED / "branch_coordinates.csv")[["branch_code", "latitude", "longitude", "coordinate_corrected"]]
table = table.merge(coords, on="branch_code", how="left", validate="1:1")
for name in FEATURE_FILES:
    table = table.merge(pd.read_csv(PROCESSED / name), on="branch_code", how="left", validate="1:1")
table["city_approx"] = table.branch_code.map(CITY_FIX).fillna(table.city_approx)
features = [f for group in FEATURE_GROUPS.values() for f in group]
table[META + [TARGET] + features].to_csv(OUT, index=False)

# validation, on the file as it will be read back
check = pd.read_csv(OUT)
assert len(check) == 127, f"expected 127 rows, got {len(check)}"
assert check.branch_code.is_unique, "duplicate branch_code"
assert len(features) == len(set(features)), "a feature name appears twice in FEATURE_GROUPS"
assert check.columns.is_unique, "duplicated column name"
assert not set(features) & set(META + [TARGET]), "metadata or target listed as a feature"
assert not LEAKY & set(check.columns), "performance column in the modeling table"
assert check[features + [TARGET, "latitude", "longitude"]].notna().all().all(), "missing values"
assert all(pd.api.types.is_numeric_dtype(check[f]) for f in features), "non-numeric feature"

print(f"saved {OUT.relative_to(ROOT)}: {check.shape[0]} rows x {check.shape[1]} columns, "
      f"{check.branch_code.nunique()} unique branch_code, 0 missing, {len(features)} feature columns")
print("features per family:", {g: len(f) for g, f in FEATURE_GROUPS.items()})
print("city_approx fixed for:", CITY_FIX)
print(f"target {TARGET}:", check[TARGET].describe().round(2).to_dict())

main = [f for g in ["demographics", "pois", "network_public", "accessibility"] for f in FEATURE_GROUPS[g]]
corr = check[main].corr(method="spearman").abs()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(main) for b in main[i + 1:] if corr.loc[a, b] >= 0.9]
print(f"\nnear-duplicate feature pairs (|Spearman| >= 0.9) among the {len(main)} main features:")
print("\n".join(f"  {a:<30} {b:<30} {r:.2f}" for a, b, r in pairs) or "  none")
print("features with <= 8 distinct values:",
      {f: int(check[f].nunique()) for f in main if check[f].nunique() <= 8})
