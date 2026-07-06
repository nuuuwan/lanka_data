import math

from shapely.geometry import Polygon
from shapely.ops import unary_union


class HexMapLabelGeoMixin:
    MERGE_EPS_FACTOR = 1e-6

    @staticmethod
    def _region_hexes(layout):
        hexes = {}
        for region_id, x, y in layout["hexes"]:
            hexes.setdefault(region_id, []).append((x, y))
        return hexes

    @staticmethod
    def _region_centroid(points):
        point_count = len(points)
        return (
            sum(x for x, _ in points) / point_count,
            sum(y for _, y in points) / point_count,
        )

    @classmethod
    def _snapped_position(cls, points):
        centroid = cls._region_centroid(points)
        return min(
            points,
            key=lambda point: (point[0] - centroid[0]) ** 2
            + (point[1] - centroid[1]) ** 2,
        )

    @classmethod
    def _region_positions(cls, layout, snap):
        positions = {}
        for region_id, points in cls._region_hexes(layout).items():
            positions[region_id] = (
                cls._snapped_position(points)
                if snap
                else cls._region_centroid(points)
            )
        return positions

    @staticmethod
    def _hex_polygon(x, y, radius):
        return Polygon(
            [
                (
                    x + radius * math.cos(math.pi / 2 + math.pi / 3 * k),
                    y + radius * math.sin(math.pi / 2 + math.pi / 3 * k),
                )
                for k in range(6)
            ]
        )

    @classmethod
    def _region_polygons(cls, layout):
        radius = layout["radius"]
        eps = cls.MERGE_EPS_FACTOR * radius
        region_to_polys = {}
        for region_id, x, y in layout["hexes"]:
            region_to_polys.setdefault(region_id, []).append(
                cls._hex_polygon(x, y, radius).buffer(eps)
            )
        return {
            region_id: unary_union(polys).buffer(-eps)
            for region_id, polys in region_to_polys.items()
        }
