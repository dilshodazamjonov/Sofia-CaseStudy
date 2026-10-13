"""Merge the stat.uz demography CSVs into one district-level reference table.

Input:  data/external/stat_uz/*.csv (from download_stat_uz.py)
Output: data/external/stat_uz/stat_demography.csv
Run: uv run python src/sofia_task/build_stat_demography.py

admin_code is an identifier: read it back with dtype={"admin_code": str} and join on it, not on admin_name.
"""
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parents[2] / "data" / "external" / "stat_uz"
YEAR = "2026"
PREV_YEAR = "2025"

data = {}
for name in ["population_density", "population_total", "population_urban", "working_age_share", "population_20_29"]:
    data[name] = pd.read_csv(DIR / f"{name}.csv", encoding="utf-8-sig", dtype={"Code": str}).set_index("Code")

names = data["population_total"]["Klassifikator_en"].str.split().str.join(" ")
names = names.str.replace(r"([a-z])(city|district)$", r"\1 \2", regex=True)  # e.g. "Khanabadcity"

total = data["population_total"][YEAR] * 1000  # source is in thousands of people
total_prev = data["population_total"][PREV_YEAR] * 1000

out = pd.DataFrame({"admin_name": names})
out["population_density_latest"] = data["population_density"][YEAR]  # people per km2
out["total_population_latest"] = total.round().astype(int)
out["population_growth_yoy"] = (total - total_prev) / total_prev
out["urban_share_latest"] = data["population_urban"][YEAR] * 1000 / total
# published only per region (4-digit code): every district gets its region's value
out["working_age_share_region_latest"] = out.index.str[:4].map(data["working_age_share"][YEAR]).to_numpy() / 100
out["age_20_29_share_latest"] = data["population_20_29"][YEAR] / total
out["reference_year"] = int(YEAR)

# keep districts and cities (7-digit codes); drop the country and region totals
out = out[out.index.str.len() == 7]
out.index.name = "admin_code"
out = out.round({
    "population_growth_yoy": 4,
    "urban_share_latest": 4,
    "working_age_share_region_latest": 4,
    "age_20_29_share_latest": 4,
})

out.to_csv(DIR / "stat_demography.csv")
print(f"saved stat_demography.csv: {len(out)} rows")
