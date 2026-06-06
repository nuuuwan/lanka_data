import geopandas

from lanka_data.where.RegionTypeUtils import RegionTypeUtils
from utils_future import WWW


class GeoDataUtils:
    @staticmethod
    def _load_raw_gdf(all_current_ids):
        region_type = RegionTypeUtils.get_region_type(all_current_ids[0])
        precision_label = {"gnd": "e3_small"}.get(region_type, "e4_medium")
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main"
            + f"/data/geo/topojson/{precision_label}/{region_type}s.topojson"
        )
        temp_topojson_file_path = WWW(url).download()
        gdf = geopandas.read_file(temp_topojson_file_path)
        return gdf[gdf["id"].isin(all_current_ids)]

    @staticmethod
    def _dissolve_by_region(gdf, region_to_current_ids):
        current_to_region = {
            current_id: region_id
            for region_id, current_ids in region_to_current_ids.items()
            for current_id in current_ids
        }
        gdf["region_id"] = gdf["id"].map(current_to_region)
        gdf = gdf.drop(columns=["id"])
        gdf_dissolved = gdf.dissolve(
            by="region_id", aggfunc="first"
        ).reset_index()
        return gdf_dissolved.rename(columns={"region_id": "id"})

    @staticmethod
    def _sort_by_region_ids(gdf, region_ids):
        gdf["id"] = gdf["id"].astype(str)
        return gdf.set_index("id").loc[region_ids].reset_index()

    @staticmethod
    def get_geopandas_dataframe(region_to_current_ids):
        all_current_ids = [
            region_id
            for region_ids in region_to_current_ids.values()
            for region_id in region_ids
        ]
        gdf = GeoDataUtils._load_raw_gdf(all_current_ids)
        gdf = GeoDataUtils._dissolve_by_region(gdf, region_to_current_ids)
        gdf = GeoDataUtils._sort_by_region_ids(
            gdf, list(region_to_current_ids.keys())
        )
        if gdf.empty:
            raise ValueError("No map data found.")
        return gdf
