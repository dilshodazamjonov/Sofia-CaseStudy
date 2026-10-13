"""Coordinate QA: compare every branch's effective coordinate with Safia's official store list, then check
the freeze criteria for the coordinate, demographics and network files.

Inputs:  data/processed/branches.csv                         (supplied coordinates, never edited)
         data/processed/branch_coordinates.csv               (effective coordinates, scripts/build_branch_coordinates.py)
         data/processed/branch_coordinate_overrides.csv      (documented corrections)
         data/external/safia/safia_locations_2026-10-11.csv  (official store list, scripts/download_safia_locations.py)
         data/processed/branch_demographics.csv, branch_network_features.csv (freeze checks only)
Output:  data/processed/branch_coordinate_qa.csv
Run: uv run python scripts/qa_branch_coordinates.py

Matching, in order: the manual decisions below, exact name, old name in brackets on the official list
("Парус (Чиланзар)" = branch "Чиланзар"), whole-word subset among stores no other branch has claimed.
Fuzzy similarity is never accepted. Distance thresholds only surface cases: coordinates change only
through branch_coordinate_overrides.csv.
"""
import re
from itertools import combinations
from pathlib import Path

import pandas as pd

from sofia_task.network_features import haversine_km

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OFFICIAL = ROOT / "data" / "external" / "safia" / "safia_locations_2026-10-11.csv"
OUT = PROCESSED / "branch_coordinate_qa.csv"
OK_M, STRONG_M, CRITICAL_M = 500, 2000, 5000  # > 5 km also covers every wrong-city case
DUPLICATE_M = 25  # two branches closer than this = copied coordinates
CITY_PREFIX = {"анд": "андижан", "нам": "наманган", "бух": "бухара"}  # "АНД Навруз", "НАМ Галаба", "БУХ Вокзал"

# manual review: name variants the rules miss -> official store (each checked on the map)
MANUAL_MATCH = {
    "W016": ("Медгородок", "МедГор is short for Медгородок"),
    "W037": ("Паркентский", "same street name, different ending"),
    "W068": ("Файзиабад", "spelling variant"),
    "W071": ("Андижан Тц", "АНД Навруз = Navruz trade centre (ТЦ) in Andijan"),
    "W092": ("Янгихаёт", "spacing variant"),
    "W157": ("Себзор", "spelling variant"),
    "W167": ("Гейдар Алиева", "case ending"),
    "W170": ("Уй базар", "spelling variant"),
    "W208": ("Бадамзар", "spelling variant"),
    "W049": ("Ташкент Сити", "City Mall: 3 official entries at Batyr Zakirov 7, exact pairing unknown; nearest shown"),
    "W124": ("S express Сити молл", "City Mall express kiosk"),
    "W222": ("S express Сити молл", "City Mall: 3 official entries at Batyr Zakirov 7, exact pairing unknown; nearest shown"),
}
# manual review: kept as supplied and flagged, because no single official store can be confirmed
FLAGGED = {
    "W061": ("ambiguous", "Турин", "branch «Нукус» is in Tashkent (W201 «Нукус-1» is the Nukus store). Supplied point is on official "
             "«Турин» (OSM Safia shop 6 m), which no branch claims; official «Глинка» on улица Нукус is 8.4 km away and also "
             "unclaimed. Needs confirmation from Safia."),
    "W081": ("ambiguous", "S express Чорсу", "supplied point is on «S express Чорсу» (Sakichmon 1/21); official «Чорсу» "
             "(Samarkand Darvoza 1) is 555 m away; neither is claimed by another branch."),
    "W093": ("ambiguous", "Интегро Чиланзар", "supplied point is on «Интегро Чиланзар» (Bunyodkor Ave 52/2); the same-name "
             "«Бунёдкор» (Chilanzar 17th quarter 33/1) is 698 m away; neither is claimed by another branch; same district."),
    "W219": ("review", "S express Максим Горький", "М.Горький = Максим Горький; the regular «Максим Горький» store is W126, so "
             "this is probably the express kiosk. Supplied point is not at any official store (nearest «IT парк» 454 m is W205). "
             "Identity inferred, not confirmed: not corrected."),
    "W158": ("unmatched", None, "no official «Северный вокзал» (official «Северный» is in Shymkent, Kazakhstan). Supplied point is "
             "86 m from Tashkent North station in OSM: plausible."),
    "W211": ("unmatched", None, "no official «Экопарк»; nearest official «Андижан Мумтоз» (873 m) is W220's store. Nothing in OSM "
             "confirms or rejects the supplied point."),
}


def norm(name):
    s = str(name).lower().replace("ё", "е")
    for uz, ru in ["қк", "ўу", "ғг", "ҳх"]:
        s = s.replace(uz, ru)
    s = re.sub(r"[`'’ʻʼ‘]", "", s)
    return " ".join(CITY_PREFIX.get(w, w) for w in re.sub(r"[^\w]+", " ", s).split())


def metres(lat, lon, lat2, lon2):
    return haversine_km(lat, lon, [lat2], [lon2])[0] * 1000


branches = pd.read_csv(PROCESSED / "branches.csv").set_index("branch_code")
coords = pd.read_csv(PROCESSED / "branch_coordinates.csv").set_index("branch_code")
overrides = pd.read_csv(PROCESSED / "branch_coordinate_overrides.csv").set_index("branch_code")
official = pd.read_csv(OFFICIAL)
official["key"] = official.name.map(lambda n: norm(re.sub(r"\(.*?\)", " ", n)))
official["old_key"] = official.name.map(lambda n: norm(m.group(1)) if (m := re.search(r"\((.*?)\)", n)) else "")
store = official.set_index("name")

# 1) match: manual decisions, then exact / old name, then whole-word subset among unclaimed stores
match = {code: (name, "manual", "medium", note) for code, (name, note) in MANUAL_MATCH.items()}
for code, b in coords.iterrows():
    if code in match or code in FLAGGED:
        continue
    key = norm(b.branch_name)
    hits = official[(official.key == key) | (official.old_key == key)]
    if len(hits) == 1:
        match[code] = (hits.name.iloc[0], "exact" if hits.key.iloc[0] == key else "old_name", "high", "")
    elif len(hits) > 1:
        match[code] = (None, "exact_multiple", "low", "several official stores: " + ", ".join(hits.name))
claimed = {m[0] for m in match.values() if m[0]}
for code, b in coords.iterrows():
    if code in match or code in FLAGGED:
        continue
    words = set(norm(b.branch_name).split())
    hits = [n for n, k in zip(official.name, official.key) if n not in claimed and (words <= set(k.split()) or set(k.split()) <= words)]
    if len(hits) == 1:
        match[code] = (hits[0], "word_subset", "medium", "")
    elif hits:
        match[code] = (None, "word_subset_multiple", "low", "several candidates: " + ", ".join(hits))

# 2) one QA row per branch
rows = []
for code, b in coords.iterrows():
    if code in FLAGGED:
        status, name, note = FLAGGED[code]
        method, confidence = "manual_review", "low"
    elif code in match:
        name, method, confidence, note = match[code]
        status = None
    else:
        name, method, confidence, note, status = None, "none", "low", "no official store matched (not reviewed)", "unmatched"
    dist = metres(b.latitude, b.longitude, store.latitude[name], store.longitude[name]) if name else None
    if status is None:
        status = "ambiguous" if name is None else "ok" if dist <= OK_M else "review" if dist <= CRITICAL_M else "critical"
        if name and STRONG_M < dist <= CRITICAL_M:
            note = "; ".join(filter(None, [note, "strong review (2-5 km)"]))
    if b.coordinate_corrected:
        moved = metres(branches.latitude[code], branches.longitude[code], b.latitude, b.longitude)
        note = "; ".join(filter(None, [note, f"coordinates corrected from the official list (moved {moved / 1000:.1f} km)"]))
    rows.append({
        "branch_code": code, "branch_name": b.branch_name,
        "original_latitude": branches.latitude[code], "original_longitude": branches.longitude[code],
        "effective_latitude": b.latitude, "effective_longitude": b.longitude, "coordinate_corrected": b.coordinate_corrected,
        "official_store_name": name, "official_address": store.address[name] if name else None,
        "official_latitude": store.latitude[name] if name else None, "official_longitude": store.longitude[name] if name else None,
        "distance_to_official_m": round(dist, 1) if dist is not None else None,
        "match_method": method, "match_confidence": confidence, "qa_status": status, "notes": note,
    })
qa = pd.DataFrame(rows)
qa.to_csv(OUT, index=False)
qa = qa.set_index("branch_code")

# 3) freeze checks
problems = []
if not (len(coords) == len(branches) == 127 and coords.index.is_unique and coords[["latitude", "longitude"]].notna().all().all()):
    problems.append("coordinate table: expected 127 unique branches with coordinates")
kept = coords.index[~coords.coordinate_corrected]
if not ((coords.latitude[kept] == branches.latitude[kept]).all() and (coords.longitude[kept] == branches.longitude[kept]).all()):
    problems.append("an uncorrected branch differs from branches.csv: rerun build_branch_coordinates.py")
if set(coords.index[coords.coordinate_corrected]) != set(overrides.index):
    problems.append("corrected branches differ from the override file: rerun build_branch_coordinates.py")
for code, o in overrides.iterrows():  # every override = supplied point -> coordinates of the official store it cites
    cited = re.search(r"«(.*?)»", o.source).group(1)
    if (branches.latitude[code], branches.longitude[code]) != (o.original_latitude, o.original_longitude):
        problems.append(f"{code}: override original coordinates differ from branches.csv")
    if (store.latitude[cited], store.longitude[cited]) != (o.corrected_latitude, o.corrected_longitude):
        problems.append(f"{code}: override does not reproduce official «{cited}»")
    if (coords.latitude[code], coords.longitude[code]) != (o.corrected_latitude, o.corrected_longitude):
        problems.append(f"{code}: branch_coordinates.csv not rebuilt")
for f in ["branch_demographics.csv", "branch_network_features.csv"]:
    if (PROCESSED / f).stat().st_mtime < (PROCESSED / "branch_coordinates.csv").stat().st_mtime:
        problems.append(f"{f} is older than branch_coordinates.csv: rerun its build script")
demo = pd.read_csv(PROCESSED / "branch_demographics.csv", dtype={"admin_code": str})
if not (len(demo) == 127 and demo.branch_code.is_unique and demo.notna().all().all()):
    problems.append("demographics: expected 127 unique branches, all matched, no missing values")
net = pd.read_csv(PROCESSED / "branch_network_features.csv")
counts = ["safia_count_500m", "safia_count_1km", "safia_count_2km", "safia_count_3km"]
distances = [c for c in net.columns if c.startswith("distance_")]
if not (len(net) == 127 and net.branch_code.is_unique and net.notna().all().all() and (net[distances] > 0).all().all()
        and all((net[a] <= net[b]).all() for a, b in zip(counts, counts[1:]))):
    problems.append("network: expected 127 unique branches, no missing values, distances > 0, monotonic counts")
duplicates = [(a, b, metres(coords.latitude[a], coords.longitude[a], coords.latitude[b], coords.longitude[b]))
              for a, b in combinations(coords.index, 2)]
duplicates = [d for d in duplicates if d[2] < DUPLICATE_M]
problems += [f"copied coordinates: {a} and {b} are {d:.1f} m apart" for a, b, d in duplicates]
problems += [f"{code}: unresolved critical ({qa.distance_to_official_m[code]:.0f} m from «{qa.official_store_name[code]}»)"
             for code in qa.index[qa.qa_status == "critical"]]
problems += [f"{code}: {qa.qa_status[code]} but not reviewed (add an override or a FLAGGED entry)"
             for code in qa.index[(qa.qa_status != "ok") & ~qa.index.isin(list(FLAGGED))]]

# 4) report
status = qa.qa_status.value_counts()
print(f"saved {OUT.relative_to(ROOT)}\n")
print(f"branches: {len(qa)}")
print(f"coordinates unchanged: {(~qa.coordinate_corrected).sum()}")
print(f"coordinates corrected: {qa.coordinate_corrected.sum()}")
for s in ["ok", "review", "critical", "ambiguous", "unmatched"]:
    print(f"qa {s}: {status.get(s, 0)}")

print("\ncorrected branches:")
for code, o in overrides.iterrows():
    moved = metres(o.original_latitude, o.original_longitude, o.corrected_latitude, o.corrected_longitude)
    print(f"  {code} {qa.branch_name[code]} | moved {moved / 1000:.1f} km | ({o.original_latitude:.6f}, {o.original_longitude:.6f}) -> "
          f"({o.corrected_latitude:.6f}, {o.corrected_longitude:.6f}) | official «{re.search(r'«(.*?)»', o.source).group(1)}»\n      {o.reason}")

print("\nremaining review / ambiguous / unmatched cases:")
for code, r in qa[qa.qa_status != "ok"].iterrows():
    where = f"«{r.official_store_name}» {r.distance_to_official_m:.0f} m" if pd.notna(r.official_store_name) else "no official store"
    print(f"  {code} {r.branch_name} [{r.qa_status}] {where}: {r.notes}")

if problems:
    print("\nNOT FROZEN:")
    for p in problems:
        print("  -", p)
else:
    print("\nfreeze checks passed: coordinates 127/127 (overrides reproducible), demographics 127 matched / 0 missing, "
          "network 127 rows / 0 missing / distances > 0 / monotonic counts, no copied coordinates, no unresolved critical cases")
    print("\nBRANCH COORDINATES FROZEN — READY FOR LOCAL POI FEATURE EXTRACTION")
