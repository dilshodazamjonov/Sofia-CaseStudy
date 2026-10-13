"""Predict the Safia sales index for one location (offline; see sofia_task.predict).

Run: uv run python scripts/predict_location.py --lat 41.3111 --lon 69.2797
Options: --features               also print the model features
         --admin-code CODE        settle an ambiguous district (the error lists the candidates)
         --exclude-store-id NAME  only to score an EXISTING branch: its own entry in safia_official_stores_uz.csv
"""
import argparse
import sys

from sofia_task.predict import limitations, location_features, ordinal, predict_location

parser = argparse.ArgumentParser(description="Predict the Safia sales index (1-100) of a location in Uzbekistan.")
parser.add_argument("--lat", type=float, required=True, help="latitude, e.g. 41.3111")
parser.add_argument("--lon", type=float, required=True, help="longitude, e.g. 69.2797")
parser.add_argument("--features", action="store_true", help="also print the model features")
parser.add_argument("--admin-code", help="stat.uz admin_code, only to settle an ambiguous district")
parser.add_argument("--exclude-store-id", help="only to score an existing branch: its own official store entry")
args = parser.parse_args()

try:
    result = predict_location(args.lat, args.lon, args.exclude_store_id, args.admin_code)
except ValueError as error:  # includes NoAdminArea and AmbiguousAdminArea
    sys.exit(f"error: {error}")

potential = round(result["location_potential"])
print(f"Predicted Safia sales index: {result['sales_index']:.1f} / 100")
print(f"Relative location potential: {potential} / 100 ({ordinal(potential)} percentile)")
print(f"Administrative area: {result['admin_name']} (stat.uz {result['admin_code']})")
if args.features:
    X, _ = location_features(args.lat, args.lon, args.exclude_store_id, args.admin_code)
    print("\nmodel features:\n" + X.T.to_string(header=False))
print("\nLimitations:")
for line in limitations():
    print(f"- {line}")
