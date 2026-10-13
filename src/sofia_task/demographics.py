"""stat.uz district demographics for any location: OSM admin area (point-in-polygon) -> stat.uz admin_code.

    import pandas as pd
    from sofia_task.demographics import FEATURES, admin_codes, load_admin_areas
    admin_areas = load_admin_areas("data/external/osm/uz_admin_areas.geojson")  # scripts/build_osm_districts.py
    stat = pd.read_csv("data/external/stat_uz/stat_demography.csv", dtype={"admin_code": str})
    hits, codes = admin_codes(41.31, 69.28, admin_areas)  # exactly one code = matched
    stat.set_index("admin_code").loc[codes[0], FEATURES]

A location is matched when the OSM areas covering it (boundary included) point to exactly one stat.uz
admin_code. Several codes have to be settled by hand: OSM has no city boundary for Termez, Urgench, Khiva
and Bayaut, so those districts' areas carry both the district and the city code (a point exactly on a
shared boundary also gets two). No code means the point is outside every mapped unit, or in a district
stat.uz does not publish yet (Yangi Toshkent, Yangi Namangan, Davlatobod). Neither case falls back to a
nearby unit. Used by scripts/build_branch_demographics.py (training) and sofia_task.predict (new sites).
"""
import json
from pathlib import Path

import shapely
from shapely.geometry import Point, shape

FEATURES = [
    "population_density_latest",
    "total_population_latest",
    "population_growth_yoy",
    "urban_share_latest",
    "working_age_share_region_latest",
    "age_20_29_share_latest",
]


def load_admin_areas(path):
    """(geojson features, their shapely geometries, an STRtree over the geometries)."""
    areas = json.loads(Path(path).read_text(encoding="utf-8"))["features"]
    geoms = [shape(a["geometry"]) for a in areas]
    return areas, geoms, shapely.STRtree(geoms)


def admin_codes(lat, lon, admin_areas):
    """Indices of the OSM areas covering the point, and the sorted stat.uz admin_codes they point to."""
    areas, _, tree = admin_areas
    hits = tree.query(Point(lon, lat), predicate="covered_by")
    return hits, sorted({c for i in hits for c in areas[i]["properties"]["admin_codes"]})
