"""OSM point-of-interest features for every branch, at the frozen (effective) branch coordinates.

Inputs:  data/processed/branch_coordinates.csv       (effective coordinates; never branches.csv)
         data/external/osm/processed/osm_pois.csv    (scripts/extract_osm_pois.py)
Output:  data/processed/branch_osm_poi_features.csv  (branch_code + POI counts only)
Run: uv run python scripts/build_branch_osm_poi_features.py

Uses sofia_task.osm_poi_features.osm_poi_features, the same function that scores a new location.
Suspicious results are printed, never fixed here.
"""
from pathlib import Path

import pandas as pd

from sofia_task.network_features import haversine_km
from sofia_task.osm_poi_features import CATEGORIES, FEATURES, add_categories, osm_poi_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "branch_osm_poi_features.csv"
NESTED = [("bakeries_500m", "bakeries_1km"), ("cafes_500m", "cafes_1km"), ("restaurants_500m", "restaurants_1km"),
          ("supermarkets_500m", "supermarkets_1km"), ("public_transport_stops_500m", "public_transport_stops_1km"),
          ("total_pois_500m", "total_pois_1km")]

branches = pd.read_csv(ROOT / "data" / "processed" / "branch_coordinates.csv")
pois = add_categories(pd.read_csv(ROOT / "data" / "external" / "osm" / "processed" / "osm_pois.csv", low_memory=False))
rows = [{"branch_code": b.branch_code, **osm_poi_features(b.latitude, b.longitude, pois)} for b in branches.itertuples()]
pd.DataFrame(rows).to_csv(OUT, index=False)

# validation, on the file as it will be read back
check = pd.read_csv(OUT)
features = list(FEATURES)
assert len(check) == len(branches) == 127, f"expected 127 rows, got {len(check)}"
assert check.branch_code.is_unique, "duplicate branch_code"
assert check[features].notna().all().all(), "missing values"
assert (check[features] >= 0).all().all(), "negative count"
for small, big in NESTED:
    assert (check[small] <= check[big]).all(), f"{small} > {big}"

names = branches.set_index("branch_code")["branch_name"]
check["branch_name"] = check.branch_code.map(names)
print(f"saved {OUT.relative_to(ROOT)}: {len(check)} rows, {check.branch_code.nunique()} unique branch_code, no missing values, all counts >= 0, 500m <= 1km")
print("\nPOIs per category in the extract (Safia's own stores excluded):")
print(pois[CATEGORIES].sum().to_string())
print("\nsummary statistics:")
print(check[features].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(1).to_string())
print("\nhighest total_pois_1km:")
print(check.nlargest(5, "total_pois_1km")[["branch_code", "branch_name", "total_pois_500m", "total_pois_1km"]].to_string(index=False))
print("\nlowest total_pois_1km:")
print(check.nsmallest(5, "total_pois_1km")[["branch_code", "branch_name", "total_pois_500m", "total_pois_1km"]].to_string(index=False))
zero = check[(check[features] == 0).all(axis=1)]
print(f"\nbranches with every POI feature = 0 ({len(zero)}):", ", ".join(zero.branch_code + " " + zero.branch_name) or "none")

# suspicious-output checks: report only
print("\nsuspicious-output checks:")
warnings = []
warnings += [f"{f}: every branch is 0" for f in features if (check[f] == 0).all()]
warnings += [f"{f}: {(check[f] >= 1000).sum()} branches with >= 1000 POIs" for f in features if f.endswith("500m") and (check[f] >= 1000).any()]
for f in features:
    share = check[f].value_counts(normalize=True).iloc[0]
    if share >= 0.9:
        warnings.append(f"{f}: {share:.0%} of branches have the same value ({check[f].mode()[0]})")
safia = pois[pois.is_safia]
self_hits = sum((haversine_km(b.latitude, b.longitude, safia.latitude, safia.longitude) <= 0.1).any() for b in branches.itertuples())
print("  " + "\n  ".join(warnings) if warnings else "  none")
print(f"  info: {self_hits} branches have a Safia-tagged OSM shop within 100 m; Safia objects are excluded, so no branch counts itself")
