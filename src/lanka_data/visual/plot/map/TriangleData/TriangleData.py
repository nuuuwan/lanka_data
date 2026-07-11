from lanka_data.visual.plot.map.GeoData.GeoData import GeoData
from lanka_data.visual.plot.map.HexData.ContiguityRepairMixin import \
    ContiguityRepairMixin
from lanka_data.visual.plot.map.HexData.GridAdjacencyMixin import \
    GridAdjacencyMixin
from lanka_data.visual.plot.map.HexData.HexDataAssignMixin import \
    HexDataAssignMixin
from lanka_data.visual.plot.map.HexData.HexDataCountMixin import \
    HexDataCountMixin
from lanka_data.visual.plot.map.TriangleData.TriangleDataCacheMixin import \
    TriangleDataCacheMixin
from lanka_data.visual.plot.map.TriangleData.TriangleDataGridMixin import \
    TriangleDataGridMixin
from utils_future import Log

log = Log("TriangleData")


class TriangleData(
    HexDataCountMixin,
    TriangleDataGridMixin,
    HexDataAssignMixin,
    GridAdjacencyMixin,
    ContiguityRepairMixin,
    TriangleDataCacheMixin,
):
    @classmethod
    def _prepare_data_list(cls, data_list):
        return data_list

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
    def _value_per_triangle_range(region_to_weight, triangles):
        region_to_count = {}
        for region_id, _, _ in triangles:
            region_to_count[region_id] = region_to_count.get(region_id, 0) + 1
        values = [
            region_to_weight[region_id] / count
            for region_id, count in region_to_count.items()
        ]
        if not values:
            return None, None
        return min(values), max(values)

    @classmethod
    def _compute(cls, data_list, region_to_weight):
        gdf = GeoData.get_geopandas_dataframe(data_list, True)
        bounds = tuple(gdf.total_bounds)
        centroids = cls._region_to_centroid(gdf)
        counts = cls.get_counts(region_to_weight)
        total_count = sum(counts.values())
        centers, size = cls.build_grid(bounds, total_count)
        triangles = cls.repair(
            cls.assign(centroids, counts, centers), centers
        )
        value_per_triangle = sum(region_to_weight.values()) / max(
            len(triangles), 1
        )
        value_min, value_max = cls._value_per_triangle_range(
            region_to_weight, triangles
        )
        return {
            "size": size,
            "origin_y": bounds[1],
            "triangles": triangles,
            "value_per_triangle": value_per_triangle,
            "value_per_triangle_min": value_min,
            "value_per_triangle_max": value_max,
        }

    @classmethod
    def get_triangle_layout(cls, data_list):
        data_list = cls._prepare_data_list(data_list)
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
