"""Map each Safia branch to its stat.uz district/city by point-in-polygon and attach demographics.

Inputs:  data/processed/branch_coordinates.csv     (scripts/build_branch_coordinates.py)
         data/external/osm/uz_admin_areas.geojson   (scripts/build_osm_districts.py)
         data/external/stat_uz/stat_demography.csv  (src/sofia_task/build_stat_demography.py)
Output:  data/processed/branch_demographics.csv
Run: uv run python scripts/build_branch_demographics.py

A branch is matched when the polygons containing it point to exactly one admin_code, or when an
ambiguous result is settled in RESOLVED (the choice must be one of the candidate codes). Unmatched and
ambiguous branches keep an empty admin_code and empty features, and are listed in the printed report.
admin_code is an identifier: read the output with dtype={"admin_code": str}.
The point-in-polygon lookup is sofia_task.demographics.admin_codes, the same function that scores a new site.
"""
import math
from pathlib import Path

import pandas as pd
import shapely
from shapely.geometry import Point

from sofia_task.demographics import FEATURES, admin_codes, load_admin_areas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "branch_demographics.csv"
NEAR_EDGE_M = 100  # report matched branches closer than this to their district boundary
# ambiguous point-in-polygon results settled by hand: branch -> one of its candidate admin_codes
RESOLVED = {
    # OSM has no Termez city boundary, so Termiz Tumani covers district + city. The branch is in the
    # city's central street grid, 0.65 km from OSM's Termez city node; nearby address: addr:city=Termiz.
    "W187": "1722401",
}


def edge_distance_m(point, polygon):
    scale = [111320 * math.cos(math.radians(point.y)), 110574]  # degrees -> metres near the point
    return shapely.transform(polygon.boundary, lambda xy: xy * scale).distance(shapely.transform(point, lambda xy: xy * scale))


branches = pd.read_csv(ROOT / "data" / "processed" / "branch_coordinates.csv")
stat = pd.read_csv(ROOT / "data" / "external" / "stat_uz" / "stat_demography.csv", dtype={"admin_code": str})
admin_areas = load_admin_areas(ROOT / "data" / "external" / "osm" / "uz_admin_areas.geojson")
areas, geoms, _ = admin_areas

rows = []

for b in branches.itertuples():
    point = Point(b.longitude, b.latitude) # type: ignore
    hits, codes = admin_codes(b.latitude, b.longitude, admin_areas)
    row = {"branch_code": b.branch_code, "lat_lon": (b.latitude, b.longitude), "osm_areas": ", ".join(areas[i]["properties"]["name"] for i in hits), "codes": codes, "resolved": False}
    if len(codes) > 1 and b.branch_code in RESOLVED:
        assert RESOLVED[b.branch_code] in codes, f"{b.branch_code}: resolved code is not one of {codes}" # type: ignore
        row.update(status="matched", admin_code=RESOLVED[b.branch_code], edge_m=None, resolved=True)
    elif len(codes) == 1:
        polygon = next(geoms[i] for i in hits if codes[0] in areas[i]["properties"]["admin_codes"])
        row.update(status="matched", admin_code=codes[0], edge_m=edge_distance_m(point, polygon))
    else:
        row.update(status="ambiguous" if codes else "unmatched", admin_code=None, edge_m=None)
    rows.append(row)
result = pd.DataFrame(rows)

out = result[["branch_code", "admin_code"]].merge(stat[["admin_code", "admin_name"] + FEATURES], on="admin_code", how="left", validate="many_to_one")
out["total_population_latest"] = out["total_population_latest"].astype("Int64")
out.to_csv(OUT, index=False)

# validation, on the file as it will be read back
check = pd.read_csv(OUT, dtype={"admin_code": str})
matched = check[check.admin_code.notna()]

assert len(check) == len(branches) == 127, f"expected 127 rows, got {len(check)}"
assert check.branch_code.is_unique, "duplicate branch_code"
assert matched[["admin_name"] + FEATURES].notna().all().all(), "a matched branch is missing a demographic feature"
assert matched.admin_code.str.fullmatch(r"\d{7}").all() and matched.admin_code.isin(stat.admin_code).all(), "bad admin_code"

names = stat.set_index("admin_code")["admin_name"]
missing = check[["admin_code", "admin_name"] + FEATURES].isna().sum().sum()
print(f"saved {OUT.relative_to(ROOT)} (admin_code read as {check.admin_code.dtype})")
print(f"rows {len(check)} | unique branch_code {check.branch_code.nunique()} | matched {(result.status == 'matched').sum()}"
      f" | ambiguous {(result.status == 'ambiguous').sum()} | unmatched {(result.status == 'unmatched').sum()} | missing demographic fields {missing}")
print("corrected coordinates used for:", ", ".join(branches.branch_code[branches.coordinate_corrected]) or "none")
part = result[result.resolved]
print(f"\nresolved by hand ({len(part)}):" + ("" if len(part) else " none"))

for r in part.itertuples():
    print(f"  {r.branch_code} {r.lat_lon} in OSM: {r.osm_areas} -> {r.admin_code} {names[r.admin_code]} (candidates: {' | '.join(r.codes)})")

for status in ["unmatched", "ambiguous"]:
    part = result[result.status == status]

    print(f"\n{status} ({len(part)}):" + ("" if len(part) else " none"))

    for r in part.itertuples():
        print(f"  {r.branch_code} {r.lat_lon} in OSM: {r.osm_areas or '-'} -> " + (" | ".join(f"{c} {names[c]}" for c in r.codes) or "no stat.uz unit"))
near = result[result.edge_m < NEAR_EDGE_M].sort_values("edge_m")
print(f"\nmatched within {NEAR_EDGE_M} m of their district boundary ({len(near)}):" + ("" if len(near) else " none"))

for r in near.itertuples():
    print(f"  {r.branch_code} {r.edge_m:.0f} m from the edge of {names[r.admin_code]}")
print("\nbranches per admin area:")
print(check.admin_name.value_counts(dropna=False).to_string())
