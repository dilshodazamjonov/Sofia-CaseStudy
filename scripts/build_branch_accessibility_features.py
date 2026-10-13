"""OSM road / accessibility features for every branch, at the frozen (effective) branch coordinates.

Inputs:  data/processed/branch_coordinates.csv            (effective coordinates; never branches.csv)
         data/external/osm/processed/osm_roads.gpkg       (scripts/extract_osm_roads.py)
Output:  data/processed/branch_accessibility_features.csv (branch_code + road features only)
Run: uv run python scripts/build_branch_accessibility_features.py

Uses sofia_task.accessibility_features.accessibility_features, the same function that scores a new location.
Suspicious results are printed, never fixed here.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyproj
from shapely.geometry import Point

from sofia_task.accessibility_features import CRS, FEATURES, MAJOR_CLASSES, TO_CRS, accessibility_features

ROOT = Path(__file__).resolve().parents[1]
ROADS = ROOT / "data" / "external" / "osm" / "processed" / "osm_roads.gpkg"
OUT = ROOT / "data" / "processed" / "branch_accessibility_features.csv"
NESTED = [("road_length_500m_km", "road_length_1km_km"), ("major_road_length_500m_km", "major_road_length_1km_km"),
          ("intersection_count_500m", "intersection_count_1km")]
MAJOR_IN_TOTAL = [("major_road_length_500m_km", "road_length_500m_km"), ("major_road_length_1km_km", "road_length_1km_km")]

branches = pd.read_csv(ROOT / "data" / "processed" / "branch_coordinates.csv")
roads = gpd.read_file(ROADS, layer="roads").to_crs(CRS)
intersections = gpd.read_file(ROADS, layer="intersections").to_crs(CRS)
rows = [{"branch_code": b.branch_code, **accessibility_features(b.latitude, b.longitude, roads, intersections)} for b in branches.itertuples()]
out = pd.DataFrame(rows).round(4)  # 0.1 m for lengths in km
out["distance_to_major_road_m"] = out["distance_to_major_road_m"].round(1)
out.to_csv(OUT, index=False)

# validation, on the file as it will be read back
check = pd.read_csv(OUT)
assert len(check) == len(branches) == 127, f"expected 127 rows, got {len(check)}"
assert check.branch_code.is_unique, "duplicate branch_code"
assert check[FEATURES].notna().all().all(), "missing values"
assert (check[FEATURES] >= 0).all().all(), "negative value"
for small, big in NESTED:
    assert (check[small] <= check[big]).all(), f"{small} > {big}"
for major, total in MAJOR_IN_TOTAL:
    assert (check[major] <= check[total]).all(), f"{major} > {total}"

names = branches.set_index("branch_code")["branch_name"]
check.insert(1, "branch_name", check.branch_code.map(names))
scale = pyproj.Proj(CRS).get_factors(branches.longitude.to_numpy(), branches.latitude.to_numpy()).meridional_scale
print(f"saved {OUT.relative_to(ROOT)}: {len(check)} rows, {check.branch_code.nunique()} unique branch_code, no missing or negative values, "
      "500m <= 1km, major road length <= road length")
print(f"{CRS} scale factor at the branches: {scale.min():.5f} to {scale.max():.5f} "
      f"(largest length/distance error {abs(scale - 1).max():.2%}, {branches.branch_code[abs(scale - 1).argmax()]})")

print("\nsummary statistics:")
print(check[FEATURES].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2).to_string())

print("\n10 branches closest to a major road:")
major = roads[roads.highway.isin(MAJOR_CLASSES)]
for r in check.nsmallest(10, "distance_to_major_road_m").itertuples():
    b = branches[branches.branch_code == r.branch_code].iloc[0]
    road = major.iloc[major.distance(Point(TO_CRS.transform(b.longitude, b.latitude))).argmin()]
    print(f"  {r.branch_code} {r.branch_name:<20} {r.distance_to_major_road_m:5.1f} m  {road.highway} «{road['name'] if pd.notna(road['name']) else 'no name'}»")

density = ["branch_code", "branch_name", "road_density_500m", "road_density_1km", "road_length_1km_km", "intersection_count_1km"]
print("\n10 highest road_density_1km (km of road per km²):")
print(check.nlargest(10, "road_density_1km")[density].to_string(index=False))
print("\n10 lowest road_density_1km:")
print(check.nsmallest(10, "road_density_1km")[density].to_string(index=False))

no_major = check[check.major_road_length_1km_km == 0]
print(f"\nbranches with zero major-road length within 1 km ({len(no_major)}):", ", ".join(no_major.branch_code + " " + no_major.branch_name) or "none")
no_major = check[check.major_road_length_500m_km == 0]
print(f"  within 500 m ({len(no_major)}):", ", ".join(f"{r.branch_code} {r.branch_name} (nearest major road {r.distance_to_major_road_m:.0f} m)"
                                                     for r in no_major.itertuples()) or "none")

# suspicious-output checks: report only
print("\nsuspicious-output checks:")
share = check.major_road_length_500m_km / check.road_length_500m_km
mostly_major = check[share > 0.8]
print(f"  major roads are > 80% of the road length within 500 m ({len(mostly_major)}):",
      ", ".join(f"{r.branch_code} {r.branch_name} ({s:.0%}, {r.intersection_count_500m} intersections)" for r, s in zip(mostly_major.itertuples(), share[mostly_major.index]))
      or "none", "-> local streets may be missing in OSM there, or the site is enclosed by major roads" if len(mostly_major) else "")
extreme = []
for f in FEATURES[1:]:  # not distance_to_major_road_m: its IQR is ~20 m, so 3 IQR flags any branch set back > 120 m from the road
    q1, q3 = check[f].quantile([0.25, 0.75])
    outside = check[(check[f] < q1 - 3 * (q3 - q1)) | (check[f] > q3 + 3 * (q3 - q1))]
    extreme += [f"{f} {r.branch_code} {r.branch_name} = {getattr(r, f)}" for r in outside.itertuples()]
print("  lengths, counts or densities more than 3 IQR outside the quartiles:", "; ".join(extreme) or "none")
