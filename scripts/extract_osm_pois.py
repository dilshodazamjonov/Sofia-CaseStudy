"""Extract OpenStreetMap points of interest once from the local PBF (no live API).

Input:  data/external/osm/uzbekistan-latest.osm.pbf
Output: data/external/osm/processed/osm_pois.csv  (one row per POI: a representative point and its tags)
Run: uv run python scripts/extract_osm_pois.py

Polygons and lines become one point inside / on them. A node is dropped as a duplicate when it lies inside
a polygon that has the same tag (e.g. both shop=supermarket) and the names match or one of them is unnamed.
is_safia marks Safia's own stores (name, brand or operator), which the feature counts leave out.
"""
import json
import re
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyrosm

from sofia_task.osm_poi_features import CATEGORIES, add_categories

ROOT = Path(__file__).resolve().parents[1]
PBF = ROOT / "data" / "external" / "osm" / "uzbekistan-latest.osm.pbf"
OUT = ROOT / "data" / "external" / "osm" / "processed" / "osm_pois.csv"
KEYS = ["shop", "amenity", "office", "tourism", "leisure", "highway", "public_transport", "railway"]
FILTER = {"shop": True, "amenity": True, "office": True, "tourism": True, "leisure": True,
          "highway": ["bus_stop"], "public_transport": ["platform", "station"], "railway": ["station", "halt"]}
SAFIA = re.compile(r"safia|сафия|сафиа", re.I)

warnings.filterwarnings("ignore")  # pyrosm / geopandas deprecation noise
raw = pyrosm.OSM(str(PBF)).get_pois(custom_filter=FILTER).reset_index(drop=True)
for k in KEYS + ["name", "operator"]:
    if k not in raw:
        raw[k] = None
extra = raw["tags"].map(lambda t: json.loads(t) if isinstance(t, str) else {})
labels = raw["name"].fillna("") + " " + raw["operator"].fillna("") + " " + extra.map(lambda t: " ".join(str(t.get(k, "")) for k in ["brand", "name:ru", "name:en", "name:uz"]))
raw["is_safia"] = labels.str.contains(SAFIA)

# node inside a polygon with the same tag and a compatible name = the same POI mapped twice
nodes = raw[raw.osm_type.eq("node")][KEYS + ["name", "geometry"]]
polygons = raw[raw.geometry.geom_type.isin(["Polygon", "MultiPolygon"])][KEYS + ["name", "geometry"]]
pairs = gpd.sjoin(nodes, polygons, predicate="within")
same_tag = pd.concat([(pairs[k + "_left"] == pairs[k + "_right"]) & pairs[k + "_left"].notna() for k in KEYS], axis=1).any(axis=1)
left, right = pairs["name_left"].fillna("").str.lower().str.strip(), pairs["name_right"].fillna("").str.lower().str.strip()
duplicates = set(pairs.index[same_tag & ((left == right) | left.eq("") | right.eq(""))])

pois = raw.drop(index=list(duplicates))
point = pois.geometry.representative_point()
out = pd.DataFrame({"osm_type": pois.osm_type, "osm_id": pois.id, "latitude": point.y, "longitude": point.x, "name": pois.name})
out[KEYS + ["is_safia"]] = pois[KEYS + ["is_safia"]]
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print(f"extracted {len(raw)} OSM objects ({raw.osm_type.value_counts().to_dict()}), dropped {len(duplicates)} node duplicates "
      f"of polygons, saved {len(out)} POIs to {OUT.relative_to(ROOT)}")
before, after = add_categories(raw), add_categories(out)
safia = out[out.is_safia]
print(f"Safia's own stores (left out of all counts): {len(safia)}")
print(pd.DataFrame({"raw objects": before[CATEGORIES].sum(), "after de-duplication": after[CATEGORIES].sum()}).to_string())
