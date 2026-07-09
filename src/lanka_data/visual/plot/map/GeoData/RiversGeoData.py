import geopandas
from shapely.geometry import shape


class RiversGeoData:
    @staticmethod
    def is_rivers(data_list):
        return bool(data_list) and all(
            d.get("region_type") == "rivers" for d in data_list
        )

    @staticmethod
    def _build_rows(data_list):
        region_ids, geometries = [], []
        for d in data_list:
            region_ids.append(d["region_id"])
            geometries.append(shape(d["geometry"]))
        return region_ids, geometries

    @classmethod
    def build(cls, data_list, enrich):
        region_ids, geometries = cls._build_rows(data_list)
        gdf = geopandas.GeoDataFrame(
            {"region_id": region_ids, "geometry": geometries},
            crs="EPSG:4326",
        )
        return enrich(gdf, data_list)
