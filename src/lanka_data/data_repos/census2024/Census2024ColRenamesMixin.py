class Census2024ColRenamesMixin:
    _COL_RENAMES: dict[str, dict[str, str]] = {
        "AgeGroup": {
            "total": "TotalPopulation",
            "age_0_14": "Age0To14",
            "age_15_59": "Age15To59",
            "age_60_64": "Age60To64",
        },
        "Gender": {
            "total": "TotalPopulation",
        },
        "Ethnicity": {
            "total": "TotalPopulation",
            "sri_lanka_tamil": "SriLankanTamil",
            "indian_tamil_or_malaiyaga_thamilar": "IndianTamil",
            "sri_lanka_moor_or_muslim": "SriLankanMoor",
            "sri_lanka_chetty": "SriLankanChetty",
            "veddhas": "Veddha",
        },
        "Religion": {
            "total": "TotalPopulation",
        },
        "Lighting": {
            "bio_gasgenerator": "BioGasGenerator",
        },
    }
