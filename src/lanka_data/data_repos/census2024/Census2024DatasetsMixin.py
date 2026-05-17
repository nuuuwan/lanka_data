class Census2024DatasetsMixin:
    _DATASETS: dict[str, str] = {
        "Housing": ("GN_housing_excel/Occupied-Housing-Units/data.tsv"),
        "AgeGroup": ("GN_population_excel/Population-by-Age-Group/data.tsv"),
        "Gender": ("GN_population_excel/Population-by-Sex/data.tsv"),
        "Households": ("HH_GND_excel/Number-of-Households/data.tsv"),
        "DrinkingWater": (
            "HH_GND_excel/Main-Source-of-Drinking-Water/data.tsv"
        ),
        "CookingFuel": (
            "HH_GND_excel"
            "/Main-Source-of-EnergyFuel-Used-for-Cooking/data.tsv"
        ),
        "Lighting": ("HH_GND_excel/Main-Source-of-Lighting/data.tsv"),
        "Toilet": ("HH_GND_excel/Toilet-Facilities/data.tsv"),
        "Ethnicity": (
            "Population-Preliminary-Report" "/Population-by-ethnicity/data.tsv"
        ),
        "Religion": (
            "Population-Preliminary-Report" "/Population-by-religion/data.tsv"
        ),
    }

    @classmethod
    def _resolve_label(cls, what_raw: str) -> str | None:
        lw = what_raw.lower()
        return next((k for k in cls._DATASETS if k.lower() == lw), None)
