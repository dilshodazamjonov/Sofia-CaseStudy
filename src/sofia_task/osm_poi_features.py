"""OpenStreetMap point-of-interest (POI) counts around any location.

    import pandas as pd
    from sofia_task.osm_poi_features import add_categories, osm_poi_features
    pois = add_categories(pd.read_csv("data/external/osm/processed/osm_pois.csv"))
    osm_poi_features(41.31, 69.28, pois)

pois: one row per real-world POI with latitude/longitude and its OSM tags (scripts/extract_osm_pois.py).
Distances are great-circle (haversine), so they are exact everywhere in Uzbekistan; a single projected
CRS would be off by up to ~0.7% at the country's edges.

Category definitions (OSM tags):
    bakery         shop = bakery | confectionery | pastry  (cake and pastry shops are direct competitors)
    cafe           amenity = cafe
    restaurant     amenity = restaurant                   (fast_food is NOT included)
    supermarket    shop = supermarket
    convenience    shop = convenience
    shopping       any shop=* except vacant/no/disused     (includes supermarkets, malls, department stores)
    school         amenity = school
    university     amenity = university | college
    healthcare     amenity = hospital | clinic | doctors   (pharmacies are NOT included)
    transport      highway = bus_stop | public_transport = platform | station | railway = station | halt
    poi            shopping, any amenity except street furniture/parking (NOT_A_PLACE), office=*,
                   visitor tourism and leisure values (TOURISM, LEISURE); transport stops are not POIs
Safia's own stores are left out of every category: they belong to the network features, and
counting them would count the branch itself.
"""
from sofia_task.network_features import haversine_km

NOT_A_SHOP = {"vacant", "no", "disused"}
NOT_A_PLACE = {  # amenity values that are infrastructure or street furniture, not places people visit
    "parking", "parking_space", "parking_entrance", "bicycle_parking", "motorcycle_parking", "bench",
    "waste_basket", "waste_disposal", "recycling", "toilets", "drinking_water", "fountain", "shelter",
    "telephone", "post_box", "letter_box", "vending_machine", "atm", "clock", "water_point", "bbq",
    "grave_yard", "hunting_stand", "loading_dock", "lounger", "watering_place", "give_box",
    "smoking_area", "charging_station", "compressed_air", "taxi",
}
TOURISM = {"hotel", "hostel", "guest_house", "motel", "apartment", "museum", "gallery", "attraction", "zoo", "theme_park", "aquarium"}
LEISURE = {"fitness_centre", "sports_centre", "water_park", "amusement_arcade", "bowling_alley", "escape_game", "dance", "ice_rink"}

FEATURES = {  # feature -> (category, radius in km)
    "bakeries_500m": ("bakery", 0.5),
    "bakeries_1km": ("bakery", 1.0),
    "cafes_500m": ("cafe", 0.5),
    "cafes_1km": ("cafe", 1.0),
    "restaurants_500m": ("restaurant", 0.5),
    "restaurants_1km": ("restaurant", 1.0),
    "supermarkets_500m": ("supermarket", 0.5),
    "supermarkets_1km": ("supermarket", 1.0),
    "convenience_stores_500m": ("convenience", 0.5),
    "shopping_pois_1km": ("shopping", 1.0),
    "schools_1km": ("school", 1.0),
    "universities_2km": ("university", 2.0),  # still extracted, but not a model input since 2026-10-13 (sofia_task.modeling)
    "healthcare_1km": ("healthcare", 1.0),
    "public_transport_stops_500m": ("transport", 0.5),
    "public_transport_stops_1km": ("transport", 1.0),
    "total_pois_500m": ("poi", 0.5),
    "total_pois_1km": ("poi", 1.0),
}
CATEGORIES = sorted({category for category, _ in FEATURES.values()})


def add_categories(pois):
    """Add one True/False column per category to the POI table (see the module docstring)."""
    pois = pois.copy()
    tag = {k: pois[k].fillna("") for k in ["shop", "amenity", "office", "tourism", "leisure", "highway", "public_transport", "railway"]}
    pois["bakery"] = tag["shop"].isin(["bakery", "confectionery", "pastry"])
    pois["cafe"] = tag["amenity"].eq("cafe")
    pois["restaurant"] = tag["amenity"].eq("restaurant")
    pois["supermarket"] = tag["shop"].eq("supermarket")
    pois["convenience"] = tag["shop"].eq("convenience")
    pois["shopping"] = tag["shop"].ne("") & ~tag["shop"].isin(NOT_A_SHOP)
    pois["school"] = tag["amenity"].eq("school")
    pois["university"] = tag["amenity"].isin(["university", "college"])
    pois["healthcare"] = tag["amenity"].isin(["hospital", "clinic", "doctors"])
    pois["transport"] = (tag["highway"].eq("bus_stop") | tag["public_transport"].isin(["platform", "station"])
                         | tag["railway"].isin(["station", "halt"]))
    pois["poi"] = (pois["shopping"] | (tag["amenity"].ne("") & ~tag["amenity"].isin(NOT_A_PLACE)) | tag["office"].ne("")
                   | tag["tourism"].isin(TOURISM) | tag["leisure"].isin(LEISURE))
    pois.loc[pois["is_safia"], CATEGORIES] = False
    return pois


def osm_poi_features(lat, lon, pois):
    """Count POIs of each category within each radius of one location. pois: output of add_categories."""
    d = haversine_km(lat, lon, pois["latitude"], pois["longitude"])
    return {name: int((pois[category] & (d <= radius)).sum()) for name, (category, radius) in FEATURES.items()}
