from matplotlib.patches import RegularPolygon

from utils_future import ColorUtils


class HexMapDrawMixin:
    DEFAULT_EDGE_WIDTH = 0.1
    IS_LIGHT_COLOR = getattr(ColorUtils, "_is_light_color")

    @classmethod
    def _default_fill_color(cls, color):
        if color and not cls.IS_LIGHT_COLOR(color):
            return "#fff"
        return "#000"

    @classmethod
    def _draw_hexes(cls, ax, layout, region_color_map):
        radius = layout["radius"]
        for region_id, x, y in layout["hexes"]:
            region_color = region_color_map.get(region_id)
            color = region_color or cls._default_fill_color(region_color)
            edge_color = "#444" if cls.IS_LIGHT_COLOR(color) else "#ccc"
            ax.add_patch(
                RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=radius,
                    orientation=0,
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=cls.DEFAULT_EDGE_WIDTH,
                )
            )
        cls._set_limits(ax, layout)

    @staticmethod
    def _set_limits(ax, layout):
        radius = layout["radius"]
        xs = [x for _, x, _ in layout["hexes"]]
        ys = [y for _, _, y in layout["hexes"]]
        if not xs:
            return
        ax.set_xlim(min(xs) - radius, max(xs) + radius)
        ax.set_ylim(min(ys) - radius, max(ys) + radius)
        ax.set_aspect("equal")
