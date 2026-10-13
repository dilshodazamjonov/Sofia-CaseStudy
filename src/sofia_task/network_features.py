"""Safia network / cannibalization features for any location.

    import pandas as pd
    from sofia_task.network_features import network_features
    branches = pd.read_csv("data/processed/branch_coordinates.csv")  # corrected branch coordinates
    network_features(41.31, 69.28, branches)  # new site: every existing branch is an "other" branch

For an existing branch, pass the reference table without that branch, so it never counts itself
(see scripts/build_branch_network_features.py).

public_network_features does the same against the official store list instead of the 127 labelled
branches (see scripts/build_branch_public_network_features.py).
"""
import numpy as np

EARTH_RADIUS_KM = 6371.0088  # mean Earth radius


def haversine_km(lat, lon, lats, lons):
    """Great-circle distance in km from one point to arrays of points."""
    lat, lon, lats, lons = map(np.radians, (lat, lon, np.asarray(lats), np.asarray(lons)))
    a = np.sin((lats - lat) / 2) ** 2 + np.cos(lat) * np.cos(lats) * np.sin((lons - lon) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def network_features(lat, lon, reference):
    """Distances to the 3 nearest reference branches and branch counts within 0.5/1/2/3 km.

    reference: DataFrame of the OTHER Safia branches, with latitude/longitude columns.
    """
    d = np.sort(haversine_km(lat, lon, reference["latitude"], reference["longitude"]))
    assert d[0] > 0, "location coincides with a reference branch: leave that branch out of reference"
    return {
        "distance_to_nearest_safia_km": float(d[0]),
        "distance_to_2nd_nearest_safia_km": float(d[1]),
        "distance_to_3rd_nearest_safia_km": float(d[2]),
        "safia_count_500m": int((d <= 0.5).sum()),
        "safia_count_1km": int((d <= 1).sum()),
        "safia_count_2km": int((d <= 2).sum()),
        "safia_count_3km": int((d <= 3).sum()),
    }


def public_network_features(lat, lon, reference_locations, exclude_store_id=None):
    """The same features against Safia's official store list (Uzbekistan only).

    reference_locations: data/processed/safia_official_stores_uz.csv (store_id, latitude, longitude).
    exclude_store_id: the location's own official store when it is one; None for a new site (nothing excluded).
    """
    reference = reference_locations[reference_locations["store_id"] != exclude_store_id]
    f = network_features(lat, lon, reference)
    return {
        "public_safia_nearest_km": f["distance_to_nearest_safia_km"],
        "public_safia_2nd_nearest_km": f["distance_to_2nd_nearest_safia_km"],
        "public_safia_3rd_nearest_km": f["distance_to_3rd_nearest_safia_km"],
        "public_safia_count_500m": f["safia_count_500m"],
        "public_safia_count_1km": f["safia_count_1km"],
        "public_safia_count_2km": f["safia_count_2km"],
        "public_safia_count_3km": f["safia_count_3km"],
    }
