"""Predict the Safia sales index for a location in Uzbekistan from its latitude and longitude.

    from sofia_task.predict import predict_location
    predict_location(41.3111, 69.2797)
    # {'sales_index': 17.104..., 'location_potential': 24.409..., 'latitude': 41.3111, 'longitude': 69.2797,
    #  'admin_code': '1726266', 'admin_name': 'Yunusabad district'}

lat/lon -> stat.uz district by point-in-polygon (sofia_task.demographics) -> 6 demographic features
        -> OSM POI counts (sofia_task.osm_poi_features.osm_poi_features)
        -> official-network features against all 160 official Uzbekistan stores, nothing excluded
           (sofia_task.network_features.public_network_features)
        -> the model's features (models/model_metadata.json, in that order)
        -> models/final_model.joblib: log1p + ExtraTrees, both fitted once at training time
        -> sales_index: the prediction clipped to the 1-100 business scale (no rescaling or calibration;
           an ExtraTrees prediction is an average of training targets, so it already lies within 1.61-86.51)
        -> location_potential: where that prediction ranks (percentile, 0-100) among the final model's
           spatial-CV out-of-fold predictions for the 127 labelled branches. A relative ranking of the
           location, not a second sales forecast; actual sales are not used.

These are the same feature functions that built the training tables (tests/test_predict.py checks all
127 branches). Everything is read from local files: no internet access is needed.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sofia_task.demographics import FEATURES as DEMOGRAPHIC_FEATURES, admin_codes, load_admin_areas
from sofia_task.network_features import haversine_km, public_network_features
from sofia_task.osm_poi_features import add_categories, osm_poi_features

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "models" / "final_model.joblib"
METADATA_FILE = ROOT / "models" / "model_metadata.json"
POIS_FILE = ROOT / "data" / "external" / "osm" / "processed" / "osm_pois.csv"        # scripts/extract_osm_pois.py
ADMIN_AREAS_FILE = ROOT / "data" / "external" / "osm" / "uz_admin_areas.geojson"     # scripts/build_osm_districts.py
STAT_FILE = ROOT / "data" / "external" / "stat_uz" / "stat_demography.csv"           # src/sofia_task/build_stat_demography.py
STORES_FILE = ROOT / "data" / "processed" / "safia_official_stores_uz.csv"           # scripts/build_branch_public_network_features.py
# reference for location_potential: the final model's spatial-CV out-of-fold predictions (scripts/train_final_model.py)
REFERENCE_FILE, REFERENCE_COLUMN = ROOT / "reports" / "final_model_oof.csv", "pred_spatial_group"
# descriptive percentile bands for display (upper bound exclusive, 100 is "High"); not business thresholds
POTENTIAL_BANDS = [(20, "Low"), (40, "Below average"), (60, "Typical"), (80, "Above average"), (101, "High")]
# the training table stores these rounded to 0.1 m (scripts/build_branch_public_network_features.py)
DISTANCES = ["public_safia_nearest_km", "public_safia_2nd_nearest_km", "public_safia_3rd_nearest_km"]
NOT_FEATURES = {"avg_sales_index", "median_sales_index", "avg_orders_index", "avg_check_index", "sales_index", "orders_index", "date"}
LIMITATIONS = [  # shown with every prediction (CLI and UI); the model metadata adds the Safia Mini one
    "Predictions are compressed: out-of-sample predictions in cross-validation spanned about {low:.0f}-{high:.0f}, while the "
    "127 labelled branches span 1.6-86.5. The model is better at flagging weak or average locations than "
    "exceptional high-sales ones.",
    "Only 127 labelled branches (101 in Tashkent city) were available for training.",
    "The target covers 10 days of sales (1-10 September 2026) of established branches.",
    "OpenStreetMap completeness differs between cities (central Nukus is mapped far more densely than "
    "Tashkent), which shifts the POI counts.",
]


class NoAdminArea(ValueError):
    """No stat.uz unit at the point, so there are no demographic features (never filled from a nearby unit)."""


class AmbiguousAdminArea(ValueError):
    """The point maps to several stat.uz units; choose one with admin_code= (training settled W187 the same way)."""

    def __init__(self, message, codes):
        super().__init__(message)
        self.codes = codes


@lru_cache(maxsize=1)
def resources():
    """Model, metadata and the frozen local reference data, loaded once per process."""
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_FILE)
    # one thread: with n_jobs=-1 the forest adds up its 500 tree predictions in thread-completion order,
    # so repeated calls could differ in the last floating-point bits (the fitted trees are unchanged)
    model.set_params(extratreesregressor__n_jobs=1)
    assert list(model.feature_names_in_) == metadata["features"], "model and metadata disagree on the features"
    assert not NOT_FEATURES & set(metadata["features"]), "a target/performance column is a model feature"
    oof = pd.read_csv(REFERENCE_FILE)
    reference = np.sort(oof[REFERENCE_COLUMN].to_numpy(float))
    assert oof.branch_code.is_unique and len(reference) == metadata["n_training_branches"] and np.isfinite(reference).all(), \
        f"{REFERENCE_FILE.name} must hold one out-of-fold prediction per labelled branch"
    return {
        "metadata": metadata,
        "model": model,
        "pois": add_categories(pd.read_csv(POIS_FILE, low_memory=False)),
        "stores": pd.read_csv(STORES_FILE),
        "admin_areas": load_admin_areas(ADMIN_AREAS_FILE),
        "stat": pd.read_csv(STAT_FILE, dtype={"admin_code": str}).set_index("admin_code"),
        "reference": reference,
    }


def limitations():
    reference = resources()["reference"]  # the out-of-fold predictions, so the range always matches the model
    return [line.format(low=reference.min(), high=reference.max()) for line in LIMITATIONS] + resources()["metadata"]["limitations"]


def location_potential(prediction):
    """Mid-rank percentile (0-100) of a predicted sales index among the 127 reference out-of-fold predictions:
    100 x (share of reference predictions below it + share at or below it) / 2."""
    reference = resources()["reference"]
    return float(100 * (np.mean(reference < prediction) + np.mean(reference <= prediction)) / 2)


def potential_band(percentile):
    """Descriptive label of a percentile, as displayed (rounded); not a business threshold."""
    return next(label for upper, label in POTENTIAL_BANDS if round(percentile) < upper)


def ordinal(n):
    return f"{n}{'th' if n % 100 in (11, 12, 13) else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def location_features(lat, lon, exclude_store_id=None, admin_code=None):
    """The model's input row for one location (1-row DataFrame, metadata feature order) and its stat.uz unit.

    exclude_store_id: only to score an existing branch - its own entry in the official store list
                      (store_id in safia_official_stores_uz.csv), as in its training row.
    admin_code: only to settle an AmbiguousAdminArea; must be one of the candidate codes.
    """
    if not (np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"not a valid coordinate: lat={lat}, lon={lon}")
    r = resources()
    stat, stores = r["stat"], r["stores"]

    hits, codes = admin_codes(lat, lon, r["admin_areas"])
    where = f"{lat}, {lon}"
    names = ", ".join(f"«{r['admin_areas'][0][i]['properties']['name']}»" for i in hits)
    if admin_code is not None:
        if admin_code not in codes:
            raise ValueError(f"admin_code {admin_code} is not a candidate at {where} (candidates: {codes or 'none'})")
    elif len(codes) == 1:
        admin_code = codes[0]
    elif codes:
        choices = ", ".join(f"{c} {stat.admin_name.get(c, '?')}" for c in codes)
        raise AmbiguousAdminArea(f"{where} lies in {names}, which covers several stat.uz units: {choices}. "
                                 f"Choose one with admin_code (CLI: --admin-code).", codes)
    elif len(hits):
        raise NoAdminArea(f"{where} lies in {names}, which has no stat.uz demographic data (a district stat.uz does not publish yet)")
    else:
        raise NoAdminArea(f"{where} is outside Uzbekistan's mapped districts (check the latitude/longitude order)")
    if admin_code not in stat.index:
        raise NoAdminArea(f"admin_code {admin_code} has no row in {STAT_FILE.name}")

    if exclude_store_id is not None and exclude_store_id not in set(stores.store_id):
        raise ValueError(f"exclude_store_id «{exclude_store_id}» is not in {STORES_FILE.name}")
    same = stores.store_id[(haversine_km(lat, lon, stores.latitude, stores.longitude) == 0) & (stores.store_id != exclude_store_id)]
    if len(same):
        raise ValueError(f"{where} is the official Safia store «{same.iloc[0]}»; to score that existing branch "
                         f"pass exclude_store_id='{same.iloc[0]}'")

    row = pd.DataFrame([{
        **osm_poi_features(lat, lon, r["pois"]),
        **public_network_features(lat, lon, stores, exclude_store_id=exclude_store_id),
        **stat.loc[admin_code, DEMOGRAPHIC_FEATURES].to_dict(),
    }])
    row[DISTANCES] = row[DISTANCES].round(4)
    features = r["metadata"]["features"]
    missing = set(features) - set(row.columns)
    assert not missing, f"features not generated: {sorted(missing)}"
    X = row[features]
    assert list(X.columns) == list(r["model"].feature_names_in_), "feature order differs from training"
    assert X.notna().all().all(), f"missing values in {X.columns[X.isna().any()].tolist()}"
    return X, {"admin_code": admin_code, "admin_name": stat.admin_name[admin_code]}


def predict_location(lat, lon, exclude_store_id=None, admin_code=None):
    """Expected sales index (1-100) of a Safia branch at lat/lon, its relative location potential
    (percentile 0-100, see location_potential) and its stat.uz unit. Values are unrounded.

    A new site needs only lat/lon; see location_features for the two optional arguments.
    """
    X, area = location_features(lat, lon, exclude_store_id, admin_code)
    prediction = float(resources()["model"].predict(X)[0])
    return {"sales_index": float(np.clip(prediction, 1, 100)), "location_potential": location_potential(prediction),
            "latitude": lat, "longitude": lon, **area}
