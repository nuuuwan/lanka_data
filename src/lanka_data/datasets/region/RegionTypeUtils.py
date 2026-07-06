from functools import cache


class RegionTypeUtils:
    _PREFIX_MAPS = {
        "LK": {
            2: "country",
            4: "province",
            5: "district",
            7: "dsd",
            10: "gnd",
        },
        "EC-": {5: "ed", 6: "pd"},
        "LG-": {8: "lg"},
    }

    @staticmethod
    def get_prefix_maps():
        return RegionTypeUtils._PREFIX_MAPS

    @staticmethod
    @cache
    def get_region_type(region_id: str) -> str:
        if "-pre" in region_id:
            region_id = region_id.split("-pre")[0]
        id_len = len(region_id)
        for prefix, id_map in RegionTypeUtils._PREFIX_MAPS.items():
            if region_id.startswith(prefix):
                region_type = id_map.get(id_len)
                if region_type:
                    return region_type
        raise ValueError(f"Invalid region ID format: {region_id}")

    @staticmethod
    def get_long_name(region_type: str) -> str:
        long_names = {
            "country": "Country",
            "province": "Province",
            "district": "District",
            "dsd": "Divisional Secretariat Division",
            "gnd": "Grama Niladhari Division",
            "ed": "Electoral District",
            "pd": "Polling Division",
            "lg": "Local Authority",
        }
        return long_names.get(region_type, region_type)

    @staticmethod
    def get_long_name_plural(region_type: str) -> str:
        if region_type in ["country", "lg"]:
            return RegionTypeUtils.get_long_name(region_type)[:-1] + "ies"
        return RegionTypeUtils.get_long_name(region_type) + "s"
