from matplotlib.patches import Circle

from utils_future import ColorUtils


class BubbleMapDrawMixin:
    BACKGROUND_COLOR = "#eee"
    BACKGROUND_EDGE_COLOR = "#ddd"
    BACKGROUND_EDGE_WIDTH = 0.4
    BUBBLE_EDGE_WIDTH = 0.5
    BUBBLE_ALPHA = 0.9
    IS_LIGHT_COLOR = getattr(ColorUtils, "_is_light_color")

    @classmethod
    def _draw_background(cls, ax, gdf_region):
        gdf_region.plot(
            ax=ax,
            color=cls.BACKGROUND_COLOR,
            edgecolor=cls.BACKGROUND_EDGE_COLOR,
            linewidth=cls.BACKGROUND_EDGE_WIDTH,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()

    @classmethod
    def _default_fill_color(cls, color):
        if color and not cls.IS_LIGHT_COLOR(color):
            return "#fff"
        return "#000"

    @classmethod
    def _draw_bubbles(cls, ax, layout, region_color_map):
        for region_id, x, y, radius in layout["bubbles"]:
            region_color = region_color_map.get(region_id)
            color = region_color or cls._default_fill_color(region_color)
            edge_color = "#444" if cls.IS_LIGHT_COLOR(color) else "#ccc"
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=radius,
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=cls.BUBBLE_EDGE_WIDTH,
                    alpha=cls.BUBBLE_ALPHA,
                )
            )
