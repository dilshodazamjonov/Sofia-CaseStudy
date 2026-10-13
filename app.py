"""Minimal UI: latitude/longitude in, expected Safia sales index out (sofia_task.predict, offline).

Run: uv run streamlit run app.py
"""
import streamlit as st

from sofia_task.demographics import admin_codes
from sofia_task.predict import limitations, location_features, ordinal, potential_band, predict_location, resources

st.set_page_config(page_title="Safia location score")
st.title("Safia location score")
st.caption("Expected sales index of a proposed Safia branch, from its coordinates only.")
lat = st.number_input("Latitude", value=41.3111, format="%.6f")
lon = st.number_input("Longitude", value=69.2797, format="%.6f")

# OSM has no city boundary for Termez, Urgench, Khiva and Bayaut, so there the district covers two
# statistical units (district and city): the user chooses, as was done by hand for the training data
admin_code = None
_, codes = admin_codes(lat, lon, resources()["admin_areas"])
if len(codes) > 1:
    names = resources()["stat"].admin_name
    admin_code = st.selectbox("This location's district covers several statistical units. Choose one:", codes,
                              format_func=lambda code: names.get(code, code))

if st.button("Predict"):
    try:
        result = predict_location(lat, lon, admin_code=admin_code)
    except ValueError as error:  # outside Uzbekistan, a district without statistics, an invalid coordinate
        st.error(str(error))
    else:
        potential = round(result["location_potential"])
        sales, rank = st.columns(2)
        sales.metric("Expected sales index", f"{result['sales_index']:.1f} / 100")
        rank.metric("Relative location potential", f"{potential} / 100")
        rank.caption(f"{ordinal(potential)} percentile · {potential_band(result['location_potential'])}")
        st.caption("The relative potential is the percentile of this prediction among the final model's out-of-fold "
                   "predictions for the 127 reference branches. It is not a second sales forecast. The label is a "
                   "descriptive band (0-19 Low, 20-39 Below average, 40-59 Typical, 60-79 Above average, "
                   "80-100 High), not a business threshold.")
        st.write(f"Administrative area: {result['admin_name']}")
        with st.expander("Model features"):
            st.dataframe(location_features(lat, lon, admin_code=admin_code)[0].T.rename(columns={0: "value"}))

st.subheader("Limitations")
st.markdown("\n".join(f"- {line}" for line in limitations()))
