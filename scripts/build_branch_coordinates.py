"""Branch coordinates used for all location features: branches.csv plus the documented overrides.

Inputs:  data/processed/branches.csv                       (supplied, never edited)
         data/processed/branch_coordinate_overrides.csv   (hand-checked corrections with reason and source)
Output:  data/processed/branch_coordinates.csv            (branch_code, branch_name, latitude, longitude, coordinate_corrected)
Run: uv run python scripts/build_branch_coordinates.py

Location features (demographics, network) read this file, never branches.csv directly.
"""
from itertools import combinations
from pathlib import Path

import pandas as pd

from sofia_task.network_features import haversine_km

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "branch_coordinates.csv"
CLOSE_M = 50  # report remaining branch pairs closer than this

branches = pd.read_csv(ROOT / "data" / "processed" / "branches.csv")
overrides = pd.read_csv(ROOT / "data" / "processed" / "branch_coordinate_overrides.csv").set_index("branch_code")

# an override applies only to the supplied coordinates it was written for
supplied = branches.set_index("branch_code").loc[overrides.index]
assert (supplied.latitude == overrides.original_latitude).all() and (supplied.longitude == overrides.original_longitude).all(), \
    "branches.csv coordinates changed under an override: review branch_coordinate_overrides.csv"

out = branches[["branch_code", "branch_name", "latitude", "longitude"]].copy()
out["coordinate_corrected"] = out.branch_code.isin(overrides.index)
fixed = out.coordinate_corrected
out.loc[fixed, "latitude"] = out.loc[fixed, "branch_code"].map(overrides.corrected_latitude)
out.loc[fixed, "longitude"] = out.loc[fixed, "branch_code"].map(overrides.corrected_longitude)
out.to_csv(OUT, index=False)

assert len(out) == len(branches) and out.branch_code.is_unique and out[["latitude", "longitude"]].notna().all().all()
print(f"saved {OUT.relative_to(ROOT)}: {len(out)} branches, {fixed.sum()} with corrected coordinates")
for code, r in overrides.iterrows():
    moved = haversine_km(r.original_latitude, r.original_longitude, [r.corrected_latitude], [r.corrected_longitude])[0]
    print(f"  {code} moved {moved:.1f} km")
print(f"branch pairs still closer than {CLOSE_M} m:")
for a, b in combinations(out.itertuples(), 2):
    d = haversine_km(a.latitude, a.longitude, [b.latitude], [b.longitude])[0] * 1000
    if d < CLOSE_M:
        print(f"  {a.branch_code} {a.branch_name} <-> {b.branch_code} {b.branch_name}: {d:.1f} m")
