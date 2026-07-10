from lanka_data.visual.plot.map.GeoData.GeoData import GeoData
from lanka_data.visual.plot.map.HexData.HexDataAssignMixin import \
    HexDataAssignMixin
from lanka_data.visual.plot.map.HexData.HexDataCacheMixin import \
    HexDataCacheMixin
from lanka_data.visual.plot.map.HexData.HexDataCountMixin import \
    HexDataCountMixin
from lanka_data.visual.plot.map.HexData.HexDataGridMixin import \
    HexDataGridMixin
from utils_future import Log

log = Log("HexData")


class HexData(
    HexDataCountMixin,
    HexDataGridMixin,
    HexDataAssignMixin,
    HexDataCacheMixin,
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

    @staticmethod
    def _value_per_hex_range(region_to_weight, hexes):
        region_to_hex_count = {}
        for region_id, _, _ in hexes:
            region_to_hex_count[region_id] = (
                region_to_hex_count.get(region_id, 0) + 1
            )
        values = [
            region_to_weight[region_id] / count
            for region_id, count in region_to_hex_count.items()
        ]
        if not values:
            return None, None
        return min(values), max(values)

    @classmethod
    def _compute(cls, data_list, region_to_weight):
        gdf = GeoData.get_geopandas_dataframe(data_list, True)
        centroids = cls._region_to_centroid(gdf)
        counts = cls.get_counts(region_to_weight)
        total_count = sum(counts.values())
        centers, radius = cls.build_grid(tuple(gdf.total_bounds), total_count)
        hexes = cls.assign(centroids, counts, centers)
        value_per_hex = sum(region_to_weight.values()) / max(len(hexes), 1)
        value_min, value_max = cls._value_per_hex_range(
            region_to_weight, hexes
        )
        return {
            "radius": radius,
            "hexes": hexes,
            "value_per_hex": value_per_hex,
            "value_per_hex_min": value_min,
            "value_per_hex_max": value_max,
        }

    @classmethod
    def get_hex_layout(cls, data_list):
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
