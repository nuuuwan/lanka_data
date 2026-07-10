from matplotlib.patches import Polygon

from lanka_data.visual.plot_visual.TriangleMapVisual.TriangleGeometry import (
    TriangleGeometry,
)
from utils_future import ColorUtils


class TriangleMapDrawMixin:
    DEFAULT_EDGE_WIDTH = 0.1
    IS_LIGHT_COLOR = getattr(ColorUtils, "_is_light_color")

    @classmethod
    def _default_fill_color(cls, color):
        if color and not cls.IS_LIGHT_COLOR(color):
            return "#fff"
        return "#000"

    @classmethod
    def _draw_triangles(cls, ax, layout, region_color_map):
        size = layout["size"]
        for region_id, x, y, up in layout["triangles"]:
            region_color = region_color_map.get(region_id)
            color = region_color or cls._default_fill_color(region_color)
            edge_color = "#444" if cls.IS_LIGHT_COLOR(color) else "#ccc"
            ax.add_patch(
                Polygon(
                    TriangleGeometry.vertices(x, y, size, up),
                    closed=True,
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=cls.DEFAULT_EDGE_WIDTH,
                )
            )
        cls._set_limits(ax, layout)

    @staticmethod
    def _set_limits(ax, layout):
        size = layout["size"]
        xs = [x for _, x, _, _ in layout["triangles"]]
        ys = [y for _, _, y, _ in layout["triangles"]]
        if not xs:
            return
        ax.set_xlim(min(xs) - size, max(xs) + size)
        ax.set_ylim(min(ys) - size, max(ys) + size)
        ax.set_aspect("equal")
