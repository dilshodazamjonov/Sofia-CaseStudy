"""Build Uzbekistan district/city boundaries tagged with stat.uz admin codes.

Source:  OpenStreetMap extract from Geofabrik (downloaded once, cached in data/external/osm/)
Input:   data/external/stat_uz/population_total.csv, population_density.csv (names and areas of stat.uz units)
Output:  data/external/osm/uz_admin_areas.geojson
Run: uv run python scripts/build_osm_districts.py

OSM boundaries carry no stat.uz (SOATO) codes, so each OSM district/city polygon is linked to an
admin_code once, here: by its Russian name (exact after normalising spelling), plus the explicit,
area-checked pairs in OSM_TO_CODE. Branches are then mapped by point-in-polygon, never by name.
A polygon gets two admin_codes when OSM does not separate a city from the district around it.
"""
import json
import math
import re
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import osmium
import pandas as pd
import shapely
from shapely import wkb
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parents[1]
STAT_DIR = ROOT / "data" / "external" / "stat_uz"
PBF_URL = "https://download.geofabrik.de/asia/uzbekistan-latest.osm.pbf"
PBF = ROOT / "data" / "external" / "osm" / "uzbekistan-latest.osm.pbf"
OUT = ROOT / "data" / "external" / "osm" / "uz_admin_areas.geojson"

# OSM relation id -> admin_code where the names are spelled differently (OSM km2 vs stat.uz km2)
OSM_TO_CODE = {
    18507227: "1727401",  # Нурафшан (Тойтепа) = Город Нурафшон            34.9 vs 32.5
    7284021: "1735401",  # Нукусский горхокимият = Город Нукус           214.4 vs 221.0
    2447560: "1726264",  # Бектимирский = Бектемирский                    35.5 vs 33.1
    1751444: "1726290",  # Яшнабадский = Яшнободский                      77.8 vs 56.7
    5745839: "1727239",  # Юкарычирчикский = Юкоричирчикский             279.9 vs 447.2
    15534554: "1703211",  # Джалакудукский = Жалакудукский               369.9 vs 368.7
    15534185: "1703209",  # Бустанский = Бустонский                      200.7 vs 200.5
    15638585: "1706246",  # Джандарский = Жондорский                    5006.8 vs 5172.5
    15634764: "1706242",  # Рамитанский = Ромитанский                   3786.6 vs 4186.0
    7593143: "1708212",  # Шараф Рашидов = Шароф-Рашидовский            1251.8 vs 1320.4
    7590755: "1708237",  # Янгиабадский = Янгиободский                   693.5 vs 721.0
    11856183: "1710237",  # Касбийский = Касбинский                      649.2 vs 734.9
    15603118: "1710240",  # Кокдалинский = Кукдалинский                 1790.2 vs 1710.9
    15598196: "1718216",  # Кошрабатский = Кошрабадский                 1920.1 vs 2160.3
    15590527: "1718227",  # Паст Даргамский = Пастдаргомский             845.9 vs 871.1
    15590675: "1718238",  # Тайлакский = Тайлякский                      242.1 vs 278.2
    11183885: "1722203",  # Бандиханский = Бандихонский                  234.8 vs 684.9
    11087239: "1724226",  # Сардобинский = Сардобский                    513.0 vs 517.5
    11086994: "1724235",  # Хавастский = Хавасский                       631.9 vs 619.7
    2094502: "1735218",  # Канликульский = Канлыкульский                 760.9 vs 743.6
}
# cities with no boundary in OSM: their place node lies inside this district polygon
ALSO_COVERS = {
    11137520: "1722401",  # Termez city inside Termiz Tumani
    2097690: "1733401",  # Urgench city inside Urganch Tumani
    2097691: "1733406",  # Khiva city inside Xiva Tumani
    11086477: "1724410",  # Shirin city inside Boyovut Tumani
}
NOT_UZBEKISTAN = {3280689}  # Zafarobod district of Tajikistan, same Russian name as the Uzbek one

LATIN_TO_CYRILLIC = str.maketrans("aceopxyABCEHKMOPTX", "асеорхуАВСЕНКМОРТХ")  # stat.uz mixes them in


def name_key(name):
    s = name.translate(LATIN_TO_CYRILLIC).lower().replace("ё", "е")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\b(город|г\.|район|шахри|туман)\b", " ", s)
    return " ".join(re.sub(r"[^\w]+", " ", s).split())


def km2(geom):
    scale = [111.32 * math.cos(math.radians(geom.centroid.y)), 110.574]  # degrees -> km near the polygon
    return shapely.transform(geom, lambda xy: xy * scale).area


stat = pd.read_csv(STAT_DIR / "population_total.csv", encoding="utf-8-sig", dtype={"Code": str}).set_index("Code")
stat = stat[stat.index.str.len() == 7]  # districts and cities
density = pd.read_csv(STAT_DIR / "population_density.csv", encoding="utf-8-sig", dtype={"Code": str}).set_index("Code")["2026"]
stat_km2 = stat["2026"] * 1000 / density[stat.index]
code_by_name = {(name_key(n), "city" if c[4] == "4" else "district"): c for c, n in stat["Klassifikator_ru"].items()}

if not PBF.exists():
    PBF.parent.mkdir(parents=True, exist_ok=True)
    request = Request(PBF_URL, headers={"User-Agent": "sofia-task"})
    with urlopen(request, timeout=600) as response, open(PBF, "wb") as f:
        shutil.copyfileobj(response, f)
reader = osmium.io.Reader(str(PBF), osmium.osm.osm_entity_bits.NOTHING)
osm_timestamp = reader.header().get("osmosis_replication_timestamp")
reader.close()

# district (admin_level 6) and Namangan city (level 5) polygons assembled from OSM boundary relations
wkb_factory = osmium.geom.WKBFactory()
processor = osmium.FileProcessor(str(PBF)).with_areas(osmium.filter.TagFilter(("boundary", "administrative")))
features = []
for area in processor.with_filter(osmium.filter.EntityFilter(osmium.osm.AREA)):
    osm_id = area.orig_id()
    if area.from_way() or area.tags.get("admin_level") not in ("5", "6") or osm_id in NOT_UZBEKISTAN:
        continue
    name_ru = area.tags.get("name:ru") or area.tags.get("name", "")
    kind = "district" if re.search(r"район|tumani|rayon", name_ru, re.I) else "city"
    if osm_id in OSM_TO_CODE:
        codes = [OSM_TO_CODE[osm_id]]
    elif (name_key(name_ru), kind) in code_by_name:
        codes = [code_by_name[(name_key(name_ru), kind)]]
    else:
        codes = []
    if osm_id in ALSO_COVERS:
        codes.append(ALSO_COVERS[osm_id])
    geom = wkb.loads(wkb_factory.create_multipolygon(area), hex=True)
    properties = {"osm_id": osm_id, "admin_level": int(area.tags["admin_level"]), "name": area.tags.get("name"), "name_ru": name_ru, "admin_codes": codes}
    features.append({"type": "Feature", "properties": properties, "geometry": mapping(geom), "_geom": geom})

# checks
found = {f["properties"]["osm_id"] for f in features}
assert set(OSM_TO_CODE) <= found and set(ALSO_COVERS) <= found, "a pinned OSM relation is missing from this snapshot"
used = [c for f in features for c in f["properties"]["admin_codes"]]
assert len(used) == len(set(used)), "an admin_code is linked to more than one OSM polygon"
linked = [f for f in features if f["properties"]["admin_codes"]]
print(f"OSM snapshot {osm_timestamp}: {len(features)} polygons, {len(linked)} linked to stat.uz codes")
missing = stat[~stat.index.isin(used)]
print(f"stat.uz units with no OSM polygon ({len(missing)}):", ", ".join(missing.index + " " + missing["Klassifikator_en"].str.strip()))
print("OSM polygons with no stat.uz unit:", ", ".join(str(f["properties"]["name"]) for f in features if not f["properties"]["admin_codes"]))
ratios = [(f["properties"]["name"], km2(f["_geom"]) / stat_km2[f["properties"]["admin_codes"][0]]) for f in linked if len(f["properties"]["admin_codes"]) == 1]
print("area OSM / stat.uz outside 0.67-1.5:", ", ".join(f"{n} {r:.2f}" for n, r in ratios if not 0.67 <= r <= 1.5))

for f in features:
    del f["_geom"]
collection = {"type": "FeatureCollection", "source": f"OpenStreetMap contributors (ODbL), {PBF_URL}", "osm_timestamp": osm_timestamp, "features": features}
OUT.write_text(json.dumps(collection, ensure_ascii=False))
print(f"saved {OUT.relative_to(ROOT)}")
