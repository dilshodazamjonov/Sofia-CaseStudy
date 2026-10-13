"""Feature groups, model pipelines and the cross-validation loop shared by the modeling scripts.

    import pandas as pd
    from sofia_task.modeling import FEATURE_GROUPS, TARGET, cross_validate, make_model
    table = pd.read_csv("data/processed/modeling_table.csv")
    folds = pd.read_csv("data/processed/cv_folds.csv")  # same row order as the table
    features = FEATURE_GROUPS["pois"] + FEATURE_GROUPS["network_public"]
    per_fold, oof = cross_validate(make_model("ridge", features), table[features], table[TARGET], folds.spatial_group)

Every transformation (log1p, scaling) is a step inside the model pipeline, so each CV fold fits it on
its own training rows only; the feature CSVs stay in raw units.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

TARGET = "avg_sales_index"  # mean daily sales index, 1-10 September 2026
META = ["branch_code", "branch_name", "format", "city_approx", "admin_name", "latitude", "longitude", "coordinate_corrected"]

# Model inputs, by family. network_public and network_127 describe the same thing against two store
# lists (official Uzbekistan list vs the 127 labelled branches): use one or the other, never both.
FEATURE_GROUPS = {
    "demographics": [
        "population_density_latest", "total_population_latest", "population_growth_yoy",
        "urban_share_latest", "working_age_share_region_latest", "age_20_29_share_latest",
    ],
    # universities_2km was removed on 2026-10-13: an audit showed it counts amenity=university|college objects
    # (mostly vocational colleges, lyceums, unnamed objects and campus parts), not universities
    "pois": [
        "bakeries_500m", "bakeries_1km", "cafes_500m", "cafes_1km", "restaurants_500m", "restaurants_1km",
        "supermarkets_500m", "supermarkets_1km", "convenience_stores_500m", "shopping_pois_1km",
        "schools_1km", "healthcare_1km",
        "public_transport_stops_500m", "public_transport_stops_1km", "total_pois_500m", "total_pois_1km",
    ],
    "network_public": [
        "public_safia_nearest_km", "public_safia_2nd_nearest_km", "public_safia_3rd_nearest_km",
        "public_safia_count_500m", "public_safia_count_1km", "public_safia_count_2km", "public_safia_count_3km",
    ],
    "network_127": [
        "distance_to_nearest_safia_km", "distance_to_2nd_nearest_safia_km", "distance_to_3rd_nearest_safia_km",
        "safia_count_500m", "safia_count_1km", "safia_count_2km", "safia_count_3km",
    ],
    "accessibility": [
        "distance_to_major_road_m", "road_density_500m", "road_density_1km",
        "major_road_length_500m_km", "major_road_length_1km_km", "intersection_count_500m", "intersection_count_1km",
    ],
}

# Right-skewed counts and distances (skew 1-8, about 0-2.5 after log1p). Demographics and road
# density/length/intersections are already roughly symmetric and stay raw.
LOG1P = FEATURE_GROUPS["pois"] + FEATURE_GROUPS["network_public"] + FEATURE_GROUPS["network_127"] + ["distance_to_major_road_m"]


def make_model(name, features):
    """A fresh, unfitted model on `features`. A "_logy" suffix fits on log1p(target) and predicts expm1."""
    log1p = ColumnTransformer([("log1p", FunctionTransformer(np.log1p), [f for f in features if f in LOG1P])],
                              remainder="passthrough")
    base = name.removesuffix("_logy")
    if base == "mean":
        model = DummyRegressor(strategy="mean")
    elif base == "median":
        model = DummyRegressor(strategy="median")
    elif base == "knn_coords":  # features = [latitude, longitude]; mean of the 10 nearest branches
        model = make_pipeline(FunctionTransformer(np.radians), KNeighborsRegressor(10, metric="haversine"))
    elif base == "ridge":  # alpha picked by efficient leave-one-out inside each training fold
        model = make_pipeline(log1p, StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    elif base == "extra_trees":  # log1p matters here: thresholds are drawn uniformly between min and max
        model = make_pipeline(log1p, ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5,
                                                         random_state=0, n_jobs=-1))
    elif base == "catboost":  # quantile split borders, so monotone transforms change nothing: raw features
        model = CatBoostRegressor(iterations=1000, depth=4, learning_rate=0.03, random_seed=0, verbose=0,
                                  allow_writing_files=False)
    else:
        raise ValueError(f"unknown model {name}")
    if name.endswith("_logy"):
        model = TransformedTargetRegressor(model, func=np.log1p, inverse_func=np.expm1)
    return model


def metrics(y, pred):
    """MAE (primary), RMSE and Spearman rank correlation; Spearman is undefined for a constant prediction."""
    return {"MAE": mean_absolute_error(y, pred), "RMSE": root_mean_squared_error(y, pred),
            "Spearman": spearmanr(y, pred).statistic if np.ptp(pred) > 0 else np.nan}


def cross_validate(model, X, y, folds):
    """Hold out each fold in turn, fit on the rest. Returns (one metrics row per fold, out-of-fold predictions)."""
    oof = pd.Series(np.nan, index=y.index)
    rows = []
    for fold in sorted(folds.unique()):
        test = (folds == fold).to_numpy()
        pred = clone(model).fit(X[~test], y[~test]).predict(X[test])
        oof[test] = pred
        rows.append({"fold": fold, "n_test": int(test.sum()), **metrics(y[test], pred)})
    return pd.DataFrame(rows), oof
