from shapely.geometry import box
from shapely.ops import unary_union


class SquareMapBoundaryMixin:
    BOUNDARY_COLOR = "#fff"
    BOUNDARY_WIDTH = 2
    MERGE_EPS_FACTOR = 1e-6

    @staticmethod
    def _square_polygon(x, y, size):
        return box(x - size, y - size, x + size, y + size)

    @classmethod
    def _region_to_polygons(cls, layout):
        size = layout["size"]
        region_to_polys = {}
        for region_id, x, y in layout["squares"]:
            region_to_polys.setdefault(region_id, []).append(
                cls._square_polygon(x, y, size)
            )
        return region_to_polys

    @classmethod
    def _merge(cls, polys, size):
        eps = cls.MERGE_EPS_FACTOR * size
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
        size = layout["size"]
        for polys in cls._region_to_polygons(layout).values():
            cls._plot_boundary(ax, cls._merge(polys, size))
