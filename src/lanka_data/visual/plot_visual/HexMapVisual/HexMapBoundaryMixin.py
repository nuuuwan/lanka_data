import math

from shapely.geometry import Polygon
from shapely.ops import unary_union


class HexMapBoundaryMixin:
    BOUNDARY_COLOR = "#fff"
    BOUNDARY_WIDTH = 2
    MERGE_EPS_FACTOR = 1e-6

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
    def _region_to_polygons(cls, layout):
        radius = layout["radius"]
        region_to_polys = {}
        for region_id, x, y in layout["hexes"]:
            region_to_polys.setdefault(region_id, []).append(
                cls._hex_polygon(x, y, radius)
            )
        return region_to_polys

    @classmethod
    def _merge(cls, polys, radius):
        eps = cls.MERGE_EPS_FACTOR * radius
        grown = unary_union([poly.buffer(eps) for poly in polys])
        return grown.buffer(-eps)

    @classmethod
    def _plot_ring(cls, ax, ring):
        xs, ys = ring.xy
        ax.plot(
            xs,
            ys,
            color=cls.BOUNDARY_COLOR,
            linewidth=cls.BOUNDARY_WIDTH,
        )

    @classmethod
    def _plot_boundary(cls, ax, geom):
        for poly in getattr(geom, "geoms", [geom]):
            if poly.is_empty:
                continue
            cls._plot_ring(ax, poly.exterior)
            for interior in poly.interiors:
                cls._plot_ring(ax, interior)

    @classmethod
    def _draw_boundaries(cls, ax, layout):
        radius = layout["radius"]
        for polys in cls._region_to_polygons(layout).values():
            cls._plot_boundary(ax, cls._merge(polys, radius))
