from functools import cache

import geopandas

from utils_future import WWW


class DistrictBackgroundGeoData:
    URL = (
        "https://raw.githubusercontent.com"
        + "/nuuuwan/lk_admin_regions/refs/heads/main"
        + "/data/geo/topojson/e4_medium/districts.topojson"
    )

    @classmethod
    @cache
    def get(cls):
        temp_path = WWW(cls.URL).download()
        gdf = geopandas.read_file(temp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        return gdf
