import geopandas
import pandas as pd

from datasets.region.RegionTypeUtils import RegionTypeUtils
from api.utils_future import WWW, timer


class GeoDataLoaderMixin:
    @timer
    @staticmethod
    def _load_raw_gdf_for_region_type(region_type, ids):
        precision_label = "e4_medium"
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main"
            + "/data/geo/topojson"
            + f"/{precision_label}/{region_type}s.topojson"
        )
        temp_topojson_file_path = WWW(url).download()
        gdf = geopandas.read_file(temp_topojson_file_path)
        gdf = gdf.rename(columns={"id": "region_id", "name": "region_name"})
        gdf = gdf[gdf["region_id"].isin(ids)]
        return gdf

    @staticmethod
    def _load_raw_gdf(all_current_ids):
        ids_by_type = {}
        for region_id in all_current_ids:
            region_type = RegionTypeUtils.get_region_type(region_id)
            ids_by_type.setdefault(region_type, []).append(region_id)
        gdfs = []
        for region_type, ids in ids_by_type.items():
            gdfs.append(
                GeoDataLoaderMixin._load_raw_gdf_for_region_type(
                    region_type, ids
                )
            )
        if len(gdfs) == 1:
            return gdfs[0]
        combined = geopandas.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        if combined.crs is None and gdfs:
            combined = combined.set_crs(gdfs[0].crs)
        return combined

    @staticmethod
    def _dissolve_by_region(gdf, region_to_current_ids):
        current_to_region = {
            current_id: region_id
            for region_id, current_ids in region_to_current_ids.items()
            for current_id in current_ids
        }
        gdf["region_id"] = gdf["region_id"].map(current_to_region)
        gdf["geometry"] = gdf["geometry"].buffer(0)
        gdf_dissolved = gdf.dissolve(
            by="region_id", aggfunc="first"
        ).reset_index()
        return gdf_dissolved.rename(columns={"region_id": "region_id"})
