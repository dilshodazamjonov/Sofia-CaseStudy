"""Safia network / cannibalization features for every branch, against the OTHER branches only.

Input:  data/processed/branch_coordinates.csv  (scripts/build_branch_coordinates.py)
Output: data/processed/branch_network_features.csv
Run: uv run python scripts/build_branch_network_features.py

Uses sofia_task.network_features.network_features, the same function that scores a new location.
"""
from pathlib import Path

import pandas as pd

from sofia_task.network_features import haversine_km, network_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "branch_network_features.csv"
DISTANCES = ["distance_to_nearest_safia_km", "distance_to_2nd_nearest_safia_km", "distance_to_3rd_nearest_safia_km"]
COUNTS = ["safia_count_500m", "safia_count_1km", "safia_count_2km", "safia_count_3km"]

branches = pd.read_csv(ROOT / "data" / "processed" / "branch_coordinates.csv")
rows = []
for b in branches.itertuples():
    others = branches[branches.branch_code != b.branch_code]  # never count the branch itself
    rows.append({"branch_code": b.branch_code, **network_features(b.latitude, b.longitude, others)})
out = pd.DataFrame(rows)
out[DISTANCES] = out[DISTANCES].round(4)  # 0.1 m
out.to_csv(OUT, index=False)

# validation, on the file as it will be read back
check = pd.read_csv(OUT)
assert len(check) == len(branches) == 127, f"expected 127 rows, got {len(check)}"
assert check.branch_code.is_unique, "duplicate branch_code"
assert check.notna().all().all(), "missing values"
assert (check[DISTANCES] > 0).all().all(), "a distance is not > 0"
assert (check[DISTANCES[0]] <= check[DISTANCES[1]]).all() and (check[DISTANCES[1]] <= check[DISTANCES[2]]).all(), "distances not ordered"
for small, big in zip(COUNTS, COUNTS[1:]):
    assert (check[small] <= check[big]).all(), f"{small} > {big}"

print(f"saved {OUT.relative_to(ROOT)}: {len(check)} rows, {check.branch_code.nunique()} unique branch_code, no missing values")
print("corrected coordinates used for:", ", ".join(branches.branch_code[branches.coordinate_corrected]) or "none")
print(check.describe().T[["mean", "min", "25%", "50%", "75%", "max"]].round(3).to_string())

print("\n10 branches with the smallest nearest-Safia distance:")
names = branches.set_index("branch_code")["branch_name"]
for r in check.nsmallest(10, DISTANCES[0]).itertuples():
    b = branches[branches.branch_code == r.branch_code].iloc[0]
    others = branches[branches.branch_code != r.branch_code]
    nearest = others.branch_code.iloc[haversine_km(b.latitude, b.longitude, others.latitude, others.longitude).argmin()]
    print(f"  {r.branch_code} {names[r.branch_code]:<20} {r.distance_to_nearest_safia_km * 1000:7.1f} m to {nearest} {names[nearest]:<20} count_500m={r.safia_count_500m}")
