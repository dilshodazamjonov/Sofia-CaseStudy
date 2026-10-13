"""Save Safia's official store list (name, address, coordinates) as a dated snapshot.

Source: https://safiabakery.uz/ru/locations (schema.org "Bakery" entries embedded in the page)
Output: data/external/safia/safia_locations_<YYYY-MM-DD>.csv
Run: uv run python scripts/download_safia_locations.py
"""
import json
import re
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

URL = "https://safiabakery.uz/ru/locations"
OUT = Path(__file__).resolve().parents[1] / "data" / "external" / "safia" / f"safia_locations_{date.today()}.csv"

with urlopen(Request(URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
    page = response.read().decode("utf-8")
items = []
for block in re.findall(r'<script type="application/ld\+json">(.*?)</script>', page, re.S):
    data = json.loads(block)
    items += data if isinstance(data, list) else [data]

rows = [
    {"name": x["name"], "address": x["address"]["streetAddress"], "latitude": x["geo"]["latitude"], "longitude": x["geo"]["longitude"], "opening_hours": x.get("openingHours")}
    for x in items if x.get("@type") == "Bakery"
]
OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"saved {len(rows)} locations to {OUT.name}")
