"""Smoke tests of the Streamlit page (app.py), run headless.

Run: uv run pytest -q
"""
from streamlit.testing.v1 import AppTest

from sofia_task.predict import ROOT, predict_location


def captions(app):
    return " ".join(c.value for c in app.caption)


def test_app_predicts_and_reports_errors():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.selectbox  # no district choice needed in central Tashkent
    app.button[0].click().run()  # default inputs: central Tashkent
    expected = predict_location(41.3111, 69.2797)
    assert app.metric[0].value == f"{expected['sales_index']:.1f} / 100"
    assert app.metric[1].value == f"{round(expected['location_potential'])} / 100"
    assert "percentile" in captions(app) and "not a second sales forecast" in captions(app)
    assert any(expected["admin_name"] in m.value for m in app.markdown)
    app.number_input[0].set_value(43.2389)
    app.number_input[1].set_value(76.8897)  # Almaty, Kazakhstan
    app.button[0].click().run()
    assert app.error and "outside Uzbekistan" in app.error[0].value
    assert not app.exception


def test_app_lets_the_user_settle_an_ambiguous_district():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.number_input[0].set_value(37.238793)
    app.number_input[1].set_value(67.280399).run()  # branch W187, Termez: OSM's Termiz Tumani = district + city
    assert app.selectbox and app.selectbox[0].options == ["Termiz district", "Termiz city"]
    app.selectbox[0].set_value("1722401")
    app.button[0].click().run()
    expected = predict_location(37.238793, 67.280399, admin_code="1722401")
    assert app.metric[0].value == f"{expected['sales_index']:.1f} / 100"
    assert any("Termiz city" in m.value for m in app.markdown)
    assert not app.error and not app.exception
