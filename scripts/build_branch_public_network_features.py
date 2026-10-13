"""Safia network features against Safia's official store list (Uzbekistan only), for every labelled branch.

Inputs:  data/processed/branch_coordinates.csv               (effective branch coordinates)
         data/processed/branch_coordinate_qa.csv             (each branch's own official store, scripts/qa_branch_coordinates.py)
         data/external/safia/safia_locations_2026-10-11.csv  (official snapshot, read only)
         data/external/osm/uz_admin_areas.geojson            (cross-check that kept stores are in Uzbekistan)
         data/processed/branch_network_features.csv          (comparison only, read only)
Outputs: data/processed/safia_official_stores_uz.csv         (the official network: reference table, also for new sites)
         data/processed/branch_public_network_features.csv
Run: uv run python scripts/build_branch_public_network_features.py

A branch never counts its own official store: the store the coordinate QA matched it to (qa_status ok);
without such a match, any official store within 100 m of the branch (these cases are reported).
"""
import json
from pathlib import Path

import pandas as pd
import shapely
from shapely.geometry import Point, shape

from sofia_task.network_features import haversine_km, public_network_features

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SNAPSHOT = ROOT / "data" / "external" / "safia" / "safia_locations_2026-10-11.csv"
STORES = PROCESSED / "safia_official_stores_uz.csv"
OUT = PROCESSED / "branch_public_network_features.csv"
FOREIGN = {"Алматы": "Kazakhstan", "Шымкент": "Kazakhstan"}  # cities on the official list outside Uzbekistan
UZ_RANGE = {"latitude": (37.1, 45.6), "longitude": (55.9, 73.2)}
SELF_M = 100  # self-location safeguard for branches without a confident official match
OLD = {  # public feature -> the same feature in the 127-branch network table
    "public_safia_nearest_km": "distance_to_nearest_safia_km",
    "public_safia_2nd_nearest_km": "distance_to_2nd_nearest_safia_km",
    "public_safia_3rd_nearest_km": "distance_to_3rd_nearest_safia_km",
    "public_safia_count_500m": "safia_count_500m",
    "public_safia_count_1km": "safia_count_1km",
    "public_safia_count_2km": "safia_count_2km",
    "public_safia_count_3km": "safia_count_3km",
}
DISTANCES, COUNTS = list(OLD)[:3], list(OLD)[3:]

# 1) official stores physically in Uzbekistan
snapshot = pd.read_csv(SNAPSHOT)
assert snapshot.name.is_unique, "official store names are used as store ids and must be unique"
city = snapshot.address.str.split(",").str[0].str.strip()
dropped = snapshot[city.isin(list(FOREIGN))]
stores = snapshot[~city.isin(list(FOREIGN))].rename(columns={"name": "store_id"})[["store_id", "address", "latitude", "longitude"]]
for col, (low, high) in UZ_RANGE.items():
    assert stores[col].between(low, high).all(), f"a kept official store has {col} outside Uzbekistan's range"
stores.to_csv(STORES, index=False)
districts = [shape(a["geometry"]) for a in json.loads((ROOT / "data" / "external" / "osm" / "uz_admin_areas.geojson").read_text(encoding="utf-8"))["features"]
             if a["properties"]["admin_codes"]]
tree = shapely.STRtree(districts)
outside = stores[[len(tree.query(Point(lon, lat), predicate="covered_by")) == 0 for lat, lon in zip(stores.latitude, stores.longitude)]]

# 2) features per branch, leaving out its own official store
branches = pd.read_csv(PROCESSED / "branch_coordinates.csv")
qa = pd.read_csv(PROCESSED / "branch_coordinate_qa.csv").set_index("branch_code")
rows, own_store, uncertain = [], {}, []
for b in branches.itertuples():
    code = b.branch_code
    if qa.qa_status[code] == "ok":
        own = qa.official_store_name[code]
        assert own in set(stores.store_id), f"{code}: its official store «{own}» is not in the Uzbekistan list"
    else:
        d = pd.Series(haversine_km(b.latitude, b.longitude, stores.latitude, stores.longitude) * 1000, index=stores.store_id)
        assert (d <= SELF_M).sum() <= 1, f"{code}: several official stores within {SELF_M} m"
        own = d.idxmin() if d.min() <= SELF_M else None
        uncertain.append((code, own, d.min()))
    own_store[code] = own
    rows.append({"branch_code": code, **public_network_features(b.latitude, b.longitude, stores, exclude_store_id=own)})
out = pd.DataFrame(rows)
out[DISTANCES] = out[DISTANCES].round(4)  # 0.1 m
out.to_csv(OUT, index=False)

# 3) validation, on the file as it will be read back
check = pd.read_csv(OUT)
assert len(check) == len(branches) == 127, f"expected 127 rows, got {len(check)}"
assert check.branch_code.is_unique, "duplicate branch_code"
assert check.notna().all().all(), "missing values"
assert (check[DISTANCES] > 0).all().all(), "a distance is not > 0"
assert (check[DISTANCES[0]] <= check[DISTANCES[1]]).all() and (check[DISTANCES[1]] <= check[DISTANCES[2]]).all(), "distances not ordered"
for small, big in zip(COUNTS, COUNTS[1:]):
    assert (check[small] <= check[big]).all(), f"{small} > {big}"

# 4) report
names = branches.set_index("branch_code")["branch_name"]
print(f"official snapshot: {len(snapshot)} stores; dropped {len(dropped)} outside Uzbekistan ("
      + ", ".join(f"{c} {n}" for c, n in city[dropped.index].value_counts().items()) + f"); kept {len(stores)} -> {STORES.relative_to(ROOT)}")
print("  kept stores outside OSM's Uzbek district outlines (kept by address):",
      ", ".join(f"«{s}» ({a})" for s, a in zip(outside.store_id, outside.address)) or "none")
print(f"saved {OUT.relative_to(ROOT)}: {len(check)} rows, {check.branch_code.nunique()} unique branch_code, no missing values, distances > 0, 500m <= 1km <= 2km <= 3km")

print("\nsummary statistics:")
print(check[list(OLD)].describe().T[["mean", "min", "25%", "50%", "75%", "max"]].round(3).to_string())

print("\n10 smallest nearest-Safia distances (official network):")
for r in check.nsmallest(10, DISTANCES[0]).itertuples():
    b = branches[branches.branch_code == r.branch_code].iloc[0]
    others = stores[stores.store_id != own_store[r.branch_code]]
    nearest = others.store_id.iloc[haversine_km(b.latitude, b.longitude, others.latitude, others.longitude).argmin()]
    print(f"  {r.branch_code} {names[r.branch_code]:<20} {r.public_safia_nearest_km * 1000:6.1f} m to «{nearest}» (own store: «{own_store[r.branch_code]}»)")

print(f"\nself-exclusion uncertain (no confident official match; excluded any official store within {SELF_M} m):")
for code, own, dist in uncertain:
    excluded = f"excluded «{own}» at {dist:.0f} m" if own else f"nothing within {SELF_M} m (nearest official store {dist:.0f} m)"
    print(f"  {code} {names[code]} [{qa.qa_status[code]}]: {excluded}. QA: {qa.notes[code]}")
shared = qa[(qa.qa_status == "ok") & qa.official_store_name.duplicated(keep=False)]
print("  same official store for several branches:", ", ".join(f"{c} -> «{s}»" for c, s in shared.official_store_name.items()) or "none",
      "(City Mall: 3 branches and 3 official entries at one site, so each branch still counts the other two)" if len(shared) else "")
missing = [c for c, own in own_store.items() if own is None]
print(f"  labelled branches with no official store at their location: {', '.join(f'{c} {names[c]}' for c in missing) or 'none'}"
      " -> as neighbours of other branches they are not in the official network")

old = pd.read_csv(PROCESSED / "branch_network_features.csv").set_index("branch_code").loc[check.branch_code]
new = check.set_index("branch_code")
changed = pd.concat([new[n] != old[o] for n, o in OLD.items()], axis=1).any(axis=1)
counts_changed = pd.concat([new[n] != old[OLD[n]] for n in COUNTS], axis=1).any(axis=1)
moved = (new[DISTANCES[0]] - old[OLD[DISTANCES[0]]]).abs() > 0.1
print(f"\ncompared with the 127-branch network features: {changed.sum()} of {len(new)} branches changed "
      f"({counts_changed.sum()} have a different count at some radius, {moved.sum()} have a nearest distance more than 100 m different)")
print(pd.DataFrame({"127-branch network": [old[OLD[n]].median() for n in OLD], "official network": [new[n].median() for n in OLD],
                    "127-branch mean": [old[OLD[n]].mean() for n in OLD], "official mean": [new[n].mean() for n in OLD]},
                   index=[n.replace("public_safia_", "") for n in OLD]).round(3).to_string())
