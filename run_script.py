import urllib.request
import json, pathlib

files = {
    "Housing": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/GN_housing_excel/Occupied-Housing-Units/data.tsv",
    "AgeGroup": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/GN_population_excel/Population-by-Age-Group/data.tsv",
    "Gender": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/GN_population_excel/Population-by-Sex/data.tsv",
    "Households": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/HH_GND_excel/Number-of-Households/data.tsv",
    "DrinkingWater": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/HH_GND_excel/Main-Source-of-Drinking-Water/data.tsv",
    "CookingFuel": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/HH_GND_excel/Main-Source-of-EnergyFuel-Used-for-Cooking/data.tsv",
    "Lighting": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/HH_GND_excel/Main-Source-of-Lighting/data.tsv",
    "Toilet": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/HH_GND_excel/Toilet-Facilities/data.tsv",
}

# Also check GIG2 ethnicity/religion TSVs
cache = pathlib.Path("/tmp/lanka_data/data_repo/index.json")
if cache.exists():
    idx = json.loads(cache.read_text())
    for key in ["populationethnicity", "populationreligion", "populationtotal", "populationagegroup", "populationgender"]:
        entries = idx.get(key, [])
        if entries:
            e = entries[0]
            files[f"GIG2:{key}:{e['year']}"] = e["url"]

for label, url in files.items():
    try:
        with urllib.request.urlopen(url) as r:
            headers = r.read().decode().splitlines()[0].split("\t")
        print(f"\n=== {label} ===")
        print(headers)
    except Exception as exc:
        print(f"\n=== {label} === ERROR: {exc}")
