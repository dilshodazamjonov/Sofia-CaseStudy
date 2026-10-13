"""OpenStreetMap road / accessibility features around any location.

    import geopandas as gpd
    from sofia_task.accessibility_features import CRS, accessibility_features
    roads_file = "data/external/osm/processed/osm_roads.gpkg"  # scripts/extract_osm_roads.py
    roads = gpd.read_file(roads_file, layer="roads").to_crs(CRS)
    intersections = gpd.read_file(roads_file, layer="intersections").to_crs(CRS)
    accessibility_features(41.31, 69.28, roads, intersections)

Road network = OSM ways with highway in ROAD_CLASSES; major roads = MAJOR_CLASSES.
Left out: *_link ramps (not in the class list; about 1.5% of the major-road length), service roads
(driveways, parking aisles), footway / path / steps / cycleway / pedestrian, track, construction.

Intersection = an OSM node where at least 3 segments of this road network meet (a way passing through
the node adds 2, a way starting or ending there adds 1): T-junctions and crossroads, not the shape
points along a road, dead ends, or a node where one way simply continues into the next.

Lengths, buffers and distances use one projected CRS for the whole country, UTM zone 42N
(EPSG:32642, central meridian 69°E, through Tashkent). Its scale error is under 0.15% at every branch
except W201 Nukus (0.7%, about 7 m per km) and grows to about 1.4% at the far western edge of Uzbekistan.

Units: distance in m, lengths in km, density in km of road per km² of buffer (length / buffer area).
A dual carriageway (a divided road mapped as two one-way ways) counts twice in the lengths and gives
2-4 intersection nodes per crossing.
"""
import pyproj
from shapely.geometry import Point

CRS = "EPSG:32642"  # UTM zone 42N
MAJOR_CLASSES = ["motorway", "trunk", "primary", "secondary"]
ROAD_CLASSES = MAJOR_CLASSES + ["tertiary", "residential", "unclassified", "living_street"]
RADII = {"500m": 500, "1km": 1000}  # metres
FEATURES = [
    "distance_to_major_road_m",
    "road_length_500m_km",
    "road_length_1km_km",
    "major_road_length_500m_km",
    "major_road_length_1km_km",
    "intersection_count_500m",
    "intersection_count_1km",
    "road_density_500m",
    "road_density_1km",
]
TO_CRS = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)


def accessibility_features(lat, lon, roads, intersections):
    """Road features of one location. roads, intersections: the two layers of osm_roads.gpkg, projected to CRS."""
    assert roads.crs == CRS and intersections.crs == CRS, f"project roads and intersections to {CRS} first"
    point = Point(TO_CRS.transform(lon, lat))
    major = roads["highway"].isin(MAJOR_CLASSES)
    f = {"distance_to_major_road_m": float(roads.geometry[major].distance(point).min())}
    for label, radius in RADII.items():
        circle = point.buffer(radius, quad_segs=64)  # 256-sided polygon, 99.99% of the circle's area
        near = roads.iloc[roads.sindex.query(circle, predicate="intersects")]
        km = near.geometry.intersection(circle).length / 1000  # length of each road inside the circle
        f[f"road_length_{label}_km"] = float(km.sum())
        f[f"major_road_length_{label}_km"] = float(km[near["highway"].isin(MAJOR_CLASSES)].sum())
        f[f"intersection_count_{label}"] = len(intersections.sindex.query(circle, predicate="intersects"))
        f[f"road_density_{label}"] = float(km.sum() / (circle.area / 1e6))
    return {name: f[name] for name in FEATURES}
