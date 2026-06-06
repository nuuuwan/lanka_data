import geopandas

from lanka_data.where.RegionTypeUtils import RegionTypeUtils
from utils_future import WWW


class GeoDataUtils:
    @staticmethod
    def _load_raw_gdf(all_current_ids):
        region_type = RegionTypeUtils.get_region_type(all_current_ids[0])
        precision_label = "e4_medium"
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
        gdf["geometry"] = gdf["geometry"].buffer(0)
        gdf_dissolved = gdf.dissolve(
            by="region_id", aggfunc="first"
        ).reset_index()
        return gdf_dissolved.rename(columns={"region_id": "id"})

    @staticmethod
    def _sort_by_region_ids(gdf, region_ids):
        gdf["id"] = gdf["id"].astype(str)
        return gdf.set_index("id").loc[region_ids].reset_index()

    @staticmethod
    def _build_region_map(data_list):
        region_to_current_ids = {}
        for d in data_list:
            region_id = d["region_id"]
            current_ids = d.get("current_ids") or [region_id]
            region_to_current_ids[region_id] = current_ids
        return region_to_current_ids

    @staticmethod
    def _enrich_from_data_list(gdf, data_list):
        import pandas as pd

        rows = []
        for d in data_list:
            row = {"id": d["region_id"], "name": d.get("region_name")}
            for k, v in d.items():
                if k in ("region_id", "region_name", "current_ids"):
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    row[k] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        overlap = [c for c in df.columns if c in gdf.columns and c != "id"]
        gdf = gdf.drop(columns=overlap)
        return gdf.merge(df, on="id", how="left")

    @staticmethod
    def get_geopandas_dataframe(data_list):
        region_to_current_ids = GeoDataUtils._build_region_map(data_list)
        all_current_ids = [
            cid
            for current_ids in region_to_current_ids.values()
            for cid in current_ids
        ]
        gdf = GeoDataUtils._load_raw_gdf(all_current_ids)
        gdf = GeoDataUtils._dissolve_by_region(gdf, region_to_current_ids)
        if gdf.empty:
            raise ValueError("No map data found.")
        return GeoDataUtils._enrich_from_data_list(gdf, data_list)
