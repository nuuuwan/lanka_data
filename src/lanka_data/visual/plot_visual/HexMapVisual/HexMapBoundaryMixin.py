import math

from shapely.geometry import Polygon
from shapely.ops import unary_union


class HexMapBoundaryMixin:
    BOUNDARY_COLOR = "#000"
    BOUNDARY_WIDTH = 1.2

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
    def _plot_boundary(cls, ax, geom):
        for poly in getattr(geom, "geoms", [geom]):
            xs, ys = poly.exterior.xy
            ax.plot(
                xs,
                ys,
                color=cls.BOUNDARY_COLOR,
                linewidth=cls.BOUNDARY_WIDTH,
            )

    @classmethod
    def _draw_boundaries(cls, ax, layout):
        for polys in cls._region_to_polygons(layout).values():
            cls._plot_boundary(ax, unary_union(polys))
