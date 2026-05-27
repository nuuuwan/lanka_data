import geopandas

from lanka_data.where.RegionTypeUtils import RegionTypeUtils
from utils_future import WWW


class RegionsGeoUtils:
    @staticmethod
    def get_geopandas_dataframe(regions):
        region_type = RegionTypeUtils.get_region_type(regions[0]["id"])
        precision_label = {"gnd": "e3_small"}.get(region_type, "e4_medium")
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main"
            + f"/data/geo/topojson/{precision_label}/{region_type}s.topojson"
        )
        temp_topojson_file_path = WWW(url).download()
        gdf_region = geopandas.read_file(temp_topojson_file_path)

        region_ids = [d["id"] for d in regions]
        gdf_region = gdf_region[gdf_region["id"].isin(region_ids)]

        if gdf_region.empty:
            raise ValueError("No map data found.")
        return gdf_region
