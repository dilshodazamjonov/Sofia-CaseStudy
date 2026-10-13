"""Download stat.uz demography CSVs into data/external/stat_uz/.

Source page: https://stat.uz/en/official-statistics/demography
Run: uv run python src/sofia_task/download_stat_uz.py
"""
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://api.siat.stat.uz/media/uploads/sdmx/sdmx_data_{}.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "external" / "stat_uz"

# file name -> SIAT dataset id (from the links on the source page)
DATASETS = {
    "population_density": 236,  # Population density
    "population_total": 246,  # Permanent population-Total
    "population_urban": 248,  # Permanent population-Urban
    "working_age_share": 3130,  # Permanent population at working age (share in total, %)
    "population_20_29": 3160,  # Population aged 20-29 (total)
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
for name, dataset_id in DATASETS.items():
    request = Request(URL.format(dataset_id), headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        (OUT_DIR / f"{name}.csv").write_bytes(response.read())
    print("saved", name)
