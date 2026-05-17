class Census2024ColRenamesMixin:
    _COL_RENAMES: dict[str, dict[str, str]] = {
        "AgeGroup": {
            "total": "total_population",
        },
        "Gender": {
            "total": "total_population",
        },
        "Ethnicity": {
            "total": "total_population",
            "sri_lanka_tamil": "sl_tamil",
            "indian_tamil_or_malaiyaga_thamilar": "indian_tamil",
            "sri_lanka_moor_or_muslim": "sl_moor",
            "sri_lanka_chetty": "sl_chetty",
            "veddhas": "veddha",
        },
        "Religion": {
            "total": "total_population",
        },
    }
