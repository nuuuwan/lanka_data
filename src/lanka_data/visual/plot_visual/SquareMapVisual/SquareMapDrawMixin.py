from matplotlib.patches import Rectangle

from utils_future import ColorUtils


class SquareMapDrawMixin:
    DEFAULT_EDGE_WIDTH = 0.1
    IS_LIGHT_COLOR = getattr(ColorUtils, "_is_light_color")

    @classmethod
    def _default_fill_color(cls, color):
        if color and not cls.IS_LIGHT_COLOR(color):
            return "#fff"
        return "#000"

    @classmethod
    def _draw_squares(cls, ax, layout, region_color_map):
        size = layout["size"]
        for region_id, x, y in layout["squares"]:
            region_color = region_color_map.get(region_id)
            color = region_color or cls._default_fill_color(region_color)
            edge_color = "#444" if cls.IS_LIGHT_COLOR(color) else "#ccc"
            ax.add_patch(
                Rectangle(
                    (x - size, y - size),
                    2 * size,
                    2 * size,
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=cls.DEFAULT_EDGE_WIDTH,
                )
            )
        cls._set_limits(ax, layout)

    @staticmethod
    def _set_limits(ax, layout):
        size = layout["size"]
        xs = [x for _, x, _ in layout["squares"]]
        ys = [y for _, _, y in layout["squares"]]
        if not xs:
            return
        ax.set_xlim(min(xs) - size, max(xs) + size)
        ax.set_ylim(min(ys) - size, max(ys) + size)
        ax.set_aspect("equal")
