from lanka_data.visual.plot.map.BubbleData.BubbleDataCacheMixin import \
    BubbleDataCacheMixin
from lanka_data.visual.plot.map.BubbleData.BubbleDataRadiusMixin import \
    BubbleDataRadiusMixin
from lanka_data.visual.plot.map.BubbleData.BubbleDataRelaxMixin import \
    BubbleDataRelaxMixin
from lanka_data.visual.plot.map.GeoData.GeoData import GeoData
from utils_future import Log

log = Log("BubbleData")


class BubbleData(
    BubbleDataRadiusMixin,
    BubbleDataRelaxMixin,
    BubbleDataCacheMixin,
):
    @staticmethod
    def _region_to_weight(data_list):
        return {
            d["region_id"]: abs(d.get("total_value", 1) or 1)
            for d in data_list
        }

    @staticmethod
    def _region_to_centroid(gdf):
        result = {}
        for _, row in gdf.iterrows():
            centroid = row["geometry"].centroid
            result[row["region_id"]] = (centroid.x, centroid.y)
        return result

    @classmethod
    def _compute(cls, data_list, region_to_weight):
        gdf = GeoData.get_geopandas_dataframe(data_list, False)
        centroids = cls._region_to_centroid(gdf)
        bounds = tuple(gdf.total_bounds)
        radii = cls.get_radii(region_to_weight, bounds)
        bubbles = cls.relax(centroids, radii, bounds)
        return {"bubbles": bubbles}

    @classmethod
    def get_bubble_layout(cls, data_list):
        region_to_weight = cls._region_to_weight(data_list)
        path = cls._cache_path(region_to_weight)
        cached = cls._load(path)
        if cached is not None:
            log.debug(f"Read {path}")
            return cached
        layout = cls._compute(data_list, region_to_weight)
        cls._save(path, layout)
        log.debug(f"Wrote {path}")
        return layout
