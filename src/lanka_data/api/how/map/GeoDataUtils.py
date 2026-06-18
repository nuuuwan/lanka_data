import hashlib
import os
import tempfile

import geopandas
import pandas as pd

from lanka_data.api.where.RegionTypeUtils import RegionTypeUtils
from utils_future import WWW, DCNUtils, Log

log = Log("GeoDataUtils")


class GeoDataUtils:
    @staticmethod
    def _load_raw_gdf(all_current_ids):
        precision_label = "e4_medium"
        ids_by_type = {}
        for region_id in all_current_ids:
            region_type = RegionTypeUtils.get_region_type(region_id)
            ids_by_type.setdefault(region_type, []).append(region_id)

        gdfs = []
        for region_type, ids in ids_by_type.items():
            url = (
                "https://raw.githubusercontent.com"
                + "/nuuuwan/lk_admin_regions/refs/heads/main"
                + "/data/geo/topojson"
                + f"/{precision_label}/{region_type}s.topojson"
            )
            temp_topojson_file_path = WWW(url).download()
            gdf = geopandas.read_file(temp_topojson_file_path)
            gdf = gdf.rename(
                columns={"id": "region_id", "name": "region_name"}
            )
            gdfs.append(gdf[gdf["region_id"].isin(ids)])

        return geopandas.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

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

    @staticmethod
    def _sort_by_region_ids(gdf, region_ids):
        gdf["region_id"] = gdf["region_id"].astype(str)
        return gdf.set_index("region_id").loc[region_ids].reset_index()

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

        rows = []
        for d in data_list:
            row = {
                "region_id": d["region_id"],
                "region_name": d.get("region_name"),
                "current_ids": d.get("current_ids", [d["region_id"]]),
            }
            for k, v in d.items():
                if k in ("region_id", "region_name", "current_ids"):
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    row[k] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        overlap = [
            c for c in df.columns if c in gdf.columns and c != "region_id"
        ]
        gdf = gdf.drop(columns=overlap)
        return gdf.merge(df, on="region_id", how="left")

    @staticmethod
    def get_temp_gdf_path(data_list, is_cartogram):
        hash = hashlib.md5(
            str(data_list).encode() + str(is_cartogram).encode()
        ).hexdigest()
        dir_temp = os.path.join(
            tempfile.gettempdir(),
            "lanka_data",
            "gdf_cache",
        )
        os.makedirs(dir_temp, exist_ok=True)
        return os.path.join(dir_temp, f"gdf_{hash}.geojson")

    @staticmethod
    def get_geopandas_dataframe(data_list, is_cartogram):
        temp_gdf_path = GeoDataUtils.get_temp_gdf_path(data_list, is_cartogram)
        if os.path.exists(temp_gdf_path):
            log.info(f"Loading GeoDataFrame from cache: {temp_gdf_path}")
            return geopandas.read_file(temp_gdf_path)
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

        if is_cartogram:
            region_id_to_weight = {
                d["region_id"]: abs(d["total_value"]) for d in data_list
            }
            gdf = DCNUtils.run_gdf(
                gdf,
                region_id_to_weight,
            )

        gnd_enriched = GeoDataUtils._enrich_from_data_list(gdf, data_list)
        geopandas.GeoDataFrame(gnd_enriched).to_file(
            temp_gdf_path, driver="GeoJSON"
        )
        log.debug(f"Wrote {temp_gdf_path}")
        return gnd_enriched
