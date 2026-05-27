import geopandas

from utils_future import WWW


class RegionsGeoMixin:
    def _get_geopandas_dataframe(self):
        region_type = self.region_type
        precision_label = {"gnd": "e3_medium"}.get(region_type, "e4_large")
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main"
            + f"/data/geo/topojson/{precision_label}/{region_type}s.topojson"
        )
        temp_topojson_file_path = WWW(url).download()
        gdf_region = geopandas.read_file(temp_topojson_file_path)

        region_ids = [d["id"] for d in self.regions]
        gdf_region = gdf_region[gdf_region["id"].isin(region_ids)]

        if gdf_region.empty:
            raise ValueError("No map data found.")
        return gdf_region
