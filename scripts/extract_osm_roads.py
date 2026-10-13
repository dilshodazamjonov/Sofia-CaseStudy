"""Extract the OpenStreetMap road network once from the local PBF (no live API).

Input:  data/external/osm/uzbekistan-latest.osm.pbf
Output: data/external/osm/processed/osm_roads.gpkg, two layers in WGS84 (EPSG:4326):
          roads          one line per OSM way with highway in ROAD_CLASSES (osm_id, highway, name)
          intersections  nodes where at least 3 road segments meet (osm_node_id, segments)
Run: uv run python scripts/extract_osm_roads.py

Segments at a node are counted over ROAD_CLASSES only: a way passing through the node adds 2, a way
starting or ending there adds 1. Definitions: sofia_task.accessibility_features.
"""
from collections import Counter
from pathlib import Path

import geopandas as gpd
import osmium
import pandas as pd
from shapely.geometry import LineString

from sofia_task.accessibility_features import CRS, MAJOR_CLASSES, ROAD_CLASSES

ROOT = Path(__file__).resolve().parents[1]
PBF = ROOT / "data" / "external" / "osm" / "uzbekistan-latest.osm.pbf"
OUT = ROOT / "data" / "external" / "osm" / "processed" / "osm_roads.gpkg"

reader = osmium.io.Reader(str(PBF), osmium.osm.osm_entity_bits.NOTHING)
osm_timestamp = reader.header().get("osmosis_replication_timestamp")
reader.close()

processor = (osmium.FileProcessor(str(PBF)).with_locations()
             .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
             .with_filter(osmium.filter.TagFilter(*[("highway", h) for h in ROAD_CLASSES])))
roads, segments = [], Counter()
for way in processor:
    last = len(way.nodes) - 1
    roads.append({"osm_id": way.id, "highway": way.tags["highway"], "name": way.tags.get("name"),
                  "geometry": LineString([(n.lon, n.lat) for n in way.nodes])})
    for i, n in enumerate(way.nodes):
        segments[n.ref] += 1 if i in (0, last) else 2
roads = gpd.GeoDataFrame(roads, crs="EPSG:4326")

segments = pd.Series(segments)
junctions = segments[segments >= 3]
location = [processor.node_location_storage.get(ref) for ref in junctions.index]  # node coordinates cached while reading
intersections = gpd.GeoDataFrame({"osm_node_id": junctions.index, "segments": junctions.to_numpy()},
                                 geometry=gpd.points_from_xy([p.lon for p in location], [p.lat for p in location]), crs="EPSG:4326")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.unlink(missing_ok=True)
roads.to_file(OUT, layer="roads", driver="GPKG")
intersections.to_file(OUT, layer="intersections", driver="GPKG")

km = roads.to_crs(CRS).length / 1000
print(f"OSM snapshot {osm_timestamp}: {len(roads)} road ways ({km.sum():,.0f} km), {len(segments)} road nodes, "
      f"{len(intersections)} intersections -> {OUT.relative_to(ROOT)}")
print(pd.DataFrame({"ways": roads.highway.value_counts(), "km": km.groupby(roads.highway).sum().round()})
      .loc[ROAD_CLASSES].assign(major=lambda t: t.index.isin(MAJOR_CLASSES)).to_string())
print("road nodes by number of segments:", segments.value_counts().sort_index().to_dict())
