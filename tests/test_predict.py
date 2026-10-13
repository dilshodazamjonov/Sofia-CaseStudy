"""Checks for the location -> sales index pipeline in sofia_task.predict.

Run: uv run pytest -q
"""
import socket

import numpy as np
import pandas as pd
import pytest

from sofia_task import predict
from sofia_task.network_features import haversine_km

PROCESSED = predict.ROOT / "data" / "processed"
TABLE = pd.read_csv(PROCESSED / "modeling_table.csv").set_index("branch_code")
QA = pd.read_csv(PROCESSED / "branch_coordinate_qa.csv").set_index("branch_code")
STORED_ADMIN = pd.read_csv(PROCESSED / "branch_demographics.csv", dtype={"admin_code": str}).set_index("branch_code").admin_code
CENTRE = (41.3111, 69.2797)  # Amir Temur square, Tashkent


def own_store(code):
    """The official store left out of a branch's training row (rule of scripts/build_branch_public_network_features.py)."""
    if QA.qa_status[code] == "ok":
        return QA.official_store_name[code]
    stores = predict.resources()["stores"]
    d = pd.Series(haversine_km(TABLE.latitude[code], TABLE.longitude[code], stores.latitude, stores.longitude) * 1000,
                  index=stores.store_id)
    return d.idxmin() if d.min() <= 100 else None


def runtime_features(code):
    """(feature row generated at runtime, whether its district had to be settled by hand) for a labelled branch."""
    lat, lon = TABLE.latitude[code], TABLE.longitude[code]
    try:
        return predict.location_features(lat, lon, exclude_store_id=own_store(code))[0], False
    except predict.AmbiguousAdminArea:  # settled by hand in training: use the code it got there
        return predict.location_features(lat, lon, exclude_store_id=own_store(code), admin_code=STORED_ADMIN[code])[0], True


# central Tashkent (0.9 km from the centre), outer Tashkent (Zangiata, 16 km out), regional (Nukus, densest OSM)
@pytest.mark.parametrize("code", ["W051", "W177", "W201"])
def test_branch_features_match_training_table(code):
    X, _ = runtime_features(code)
    stored = TABLE.loc[[code], X.columns]
    np.testing.assert_allclose(X.to_numpy(float), stored.to_numpy(float), rtol=0, atol=1e-9)
    model = predict.resources()["model"]
    assert model.predict(X)[0] == model.predict(stored)[0]


def test_all_127_branches_match_training_table():
    rows, settled = [], []
    for code in TABLE.index:
        X, by_hand = runtime_features(code)
        rows.append(X)
        settled += [code] * by_hand
    runtime = pd.concat(rows).set_index(TABLE.index)
    stored = TABLE[runtime.columns]
    diff = (runtime.astype(float) - stored.astype(float)).abs()
    assert diff.max().max() <= 1e-9, f"largest difference: {diff.max().idxmax()} {diff.max().max()}"
    model = predict.resources()["model"]
    assert (model.predict(runtime) == model.predict(stored)).all()
    assert settled == ["W187"]  # the one branch whose district training settled by hand (Termez)


def test_feature_schema():
    metadata, model = predict.resources()["metadata"], predict.resources()["model"]
    features, families = metadata["features"], metadata["feature_families"]
    assert len(features) == len(set(features)) == 22
    assert "universities_2km" not in features  # removed 2026-10-13 (audit: counts colleges/lyceums, not universities)
    assert list(model.feature_names_in_) == features
    assert [len(families[g]) for g in ["pois", "network_public", "demographics"]] == [9, 7, 6]
    assert not [f for f in families["pois"] if f.endswith("_500m")]
    assert "public_safia_count_500m" in features  # a network feature, kept
    assert not predict.NOT_FEATURES & set(features)
    X, _ = predict.location_features(*CENTRE)
    assert list(X.columns) == features and X.shape == (1, 22) and X.notna().all().all()


def test_prediction_is_finite_in_range_and_offline(monkeypatch):
    def no_network(*args, **kwargs):
        raise OSError("network access attempted")
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    predict.resources.cache_clear()  # load the model and every data file with the network blocked as well
    for lat, lon in [CENTRE, (39.6542, 66.9597), (40.7821, 72.3442)]:  # Tashkent, Samarkand, Andijan
        score = predict.predict_location(lat, lon)["sales_index"]
        assert np.isfinite(score) and 1 <= score <= 100


def test_same_location_gives_the_same_result():
    first = predict.predict_location(*CENTRE)
    assert predict.predict_location(*CENTRE) == first
    predict.resources.cache_clear()  # fresh load of the model and the data
    assert predict.predict_location(*CENTRE) == first


def test_outside_uzbekistan_is_an_error_not_a_guess():
    with pytest.raises(predict.NoAdminArea):
        predict.predict_location(43.2389, 76.8897)  # Almaty, Kazakhstan


def test_ambiguous_district_must_be_settled():
    lat, lon = TABLE.latitude["W187"], TABLE.longitude["W187"]  # Termez: OSM's Termiz Tumani covers district + city
    with pytest.raises(predict.AmbiguousAdminArea) as error:
        predict.predict_location(lat, lon)
    assert error.value.codes == ["1722220", "1722401"]
    assert predict.predict_location(lat, lon, admin_code="1722401")["admin_name"] == "Termiz city"


def test_existing_store_location_needs_its_store_excluded():
    stores = predict.resources()["stores"].set_index("store_id")
    lat, lon = stores.latitude["IT парк"], stores.longitude["IT парк"]
    with pytest.raises(ValueError, match="exclude_store_id"):
        predict.predict_location(lat, lon)
    assert 1 <= predict.predict_location(lat, lon, exclude_store_id="IT парк")["sales_index"] <= 100


# --- relative location potential (a percentile of the prediction, presentation layer only) ---
NEW_SITES = [CENTRE, (39.6542, 66.9597), (40.7821, 72.3442),  # Tashkent, Samarkand, Andijan
             (39.7747, 64.4286), (40.9983, 71.6726), (40.3864, 71.7864), (41.4689, 69.5822)]  # Bukhara, Namangan, Fergana, Chirchiq
# sales_index of the first three sites from the final 22-feature model (2026-10-13, universities_2km removed);
# the potential layer must not change it, and any change to model, features or preprocessing must update it deliberately
SALES_INDEX_BEFORE = [16.927151452380954, 20.16281012698412, 17.490487095238088]


def test_potential_is_a_finite_percentile():
    for lat, lon in NEW_SITES:
        potential = predict.predict_location(lat, lon)["location_potential"]
        assert np.isfinite(potential) and 0 <= potential <= 100
    reference = predict.resources()["reference"]
    assert predict.location_potential(reference.min() - 1) == 0
    assert predict.location_potential(reference.max() + 1) == 100


def test_potential_never_decreases_as_the_sales_index_rises():
    grid = [predict.location_potential(x) for x in np.linspace(0, 100, 2001)]
    assert (np.diff(grid) >= 0).all()
    results = sorted((predict.predict_location(lat, lon) for lat, lon in NEW_SITES), key=lambda r: r["sales_index"])
    assert all(a["location_potential"] <= b["location_potential"] for a, b in zip(results, results[1:]))


def test_identical_inputs_give_the_identical_potential():
    first = predict.predict_location(*CENTRE)["location_potential"]
    assert predict.predict_location(*CENTRE)["location_potential"] == first
    predict.resources.cache_clear()
    assert predict.predict_location(*CENTRE)["location_potential"] == first


def test_reference_is_the_127_final_out_of_fold_predictions():
    oof = pd.read_csv(predict.REFERENCE_FILE)
    reference = predict.resources()["reference"]
    assert len(reference) == 127 and sorted(oof.branch_code) == sorted(TABLE.index)
    np.testing.assert_array_equal(reference, np.sort(oof[predict.REFERENCE_COLUMN].to_numpy(float)))
    # they come from the frozen final model: their fold-mean spatial MAE is the one in its metadata
    mae = (oof[predict.REFERENCE_COLUMN] - oof.avg_sales_index).abs().groupby(oof.spatial_group).mean().mean()
    assert abs(mae - predict.resources()["metadata"]["validation"]["spatial"]["MAE_mean"]) < 0.001


def test_sales_index_is_unchanged_by_the_potential_layer():
    for (lat, lon), before in zip(NEW_SITES, SALES_INDEX_BEFORE):
        assert predict.predict_location(lat, lon)["sales_index"] == pytest.approx(before, rel=1e-12)
