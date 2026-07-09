from functools import cache

from utils_future import WWW

RIVER_ID_PREFIX = "R-"


class RiversData:
    URL = (
        "https://raw.githubusercontent.com"
        + "/nuuuwan/lk_rivers/refs/heads/main"
        + "/data/sri_lanka_rivers.geojson"
    )

    @staticmethod
    def _segment_lines(geometry):
        if geometry["type"] == "MultiLineString":
            return geometry["coordinates"]
        return [geometry["coordinates"]]

    @classmethod
    def _group_by_main_river(cls, features):
        main_river_to_lines = {}
        for feature in features:
            main_river_id = feature["properties"]["MAIN_RIV"]
            lines = main_river_to_lines.setdefault(main_river_id, [])
            lines.extend(cls._segment_lines(feature["geometry"]))
        return main_river_to_lines

    @staticmethod
    def _build_region(main_river_id, lines):
        return {
            "region_id": f"{RIVER_ID_PREFIX}{main_river_id}",
            "region_name": f"River {main_river_id}",
            "region_type": "rivers",
            "geometry": {
                "type": "MultiLineString",
                "coordinates": lines,
            },
        }

    @classmethod
    @cache
    def get_river_regions(cls):
        data = WWW(cls.URL).read_json()
        main_river_to_lines = cls._group_by_main_river(data["features"])
        return [
            cls._build_region(main_river_id, lines)
            for main_river_id, lines in main_river_to_lines.items()
        ]
