from functools import cache

from utils_future import WWW

from lanka_data.datasets.region.rivers.RiverNames import RiverNames

RIVER_ID_PREFIX = "R-"

LABEL_RIVER_LEN = "RiverLen"
LABEL_CATCHMENT = "Catchment"


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

    @staticmethod
    def _centroid(lines):
        lats, lngs = [], []
        for line in lines:
            for lng, lat in line:
                lats.append(lat)
                lngs.append(lng)
        n = len(lats)
        return sum(lats) / n, sum(lngs) / n

    @classmethod
    def _group_by_main_river(cls, features):
        main_river_to_lines = {}
        for feature in features:
            main_river_id = feature["properties"]["MAIN_RIV"]
            lines = main_river_to_lines.setdefault(main_river_id, [])
            lines.extend(cls._segment_lines(feature["geometry"]))
        return main_river_to_lines

    @classmethod
    def _build_region(cls, main_river_id, lines):
        center_lat, center_lng = cls._centroid(lines)
        return {
            "region_id": f"{RIVER_ID_PREFIX}{main_river_id}",
            "region_name": RiverNames.get_name(main_river_id),
            "region_type": "rivers",
            "center_lat": center_lat,
            "center_lng": center_lng,
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

    @classmethod
    def _set_outlet_measures(cls, measures, feature):
        properties = feature["properties"]
        if properties["HYRIV_ID"] != properties["MAIN_RIV"]:
            return
        region_id = f"{RIVER_ID_PREFIX}{properties['MAIN_RIV']}"
        measures[region_id] = {
            LABEL_RIVER_LEN: properties.get("DIST_UP_KM") or 0,
            LABEL_CATCHMENT: properties.get("UPLAND_SKM") or 0,
        }

    @classmethod
    @cache
    def get_river_measures(cls):
        data = WWW(cls.URL).read_json()
        measures = {}
        for feature in data["features"]:
            cls._set_outlet_measures(measures, feature)
        return measures
