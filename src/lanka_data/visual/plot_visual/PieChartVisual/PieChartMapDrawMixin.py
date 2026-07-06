from matplotlib.patches import Wedge


class PieChartMapDrawMixin:
    BACKGROUND_COLOR = "#eee"
    BACKGROUND_EDGE_COLOR = "#ddd"
    BACKGROUND_EDGE_WIDTH = 0.4
    WEDGE_EDGE_COLOR = "white"
    WEDGE_EDGE_WIDTH = 0.2

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

    @staticmethod
    def _positions(layout):
        return {
            region_id: (x, y, radius)
            for region_id, x, y, radius in layout["bubbles"]
        }

    @classmethod
    def _draw_one_pie(cls, ax, center, radius, ordered, category_to_color):
        total = sum(v for _, v in ordered) or 1
        start = 90.0
        for label, value in ordered:
            extent = 360.0 * value / total
            ax.add_patch(
                Wedge(
                    center,
                    radius,
                    start - extent,
                    start,
                    facecolor=category_to_color.get(label, "#999"),
                    edgecolor=cls.WEDGE_EDGE_COLOR,
                    linewidth=cls.WEDGE_EDGE_WIDTH,
                )
            )
            start -= extent

    @classmethod
    def _draw_map_pies(cls, ax, subregions, positions, category_to_color):
        for subregion in subregions:
            pos = positions.get(subregion["region_id"])
            if pos is None:
                continue
            ordered = cls._order_positive_values_with_top_first(
                subregion["values"]
            )
            if not ordered:
                continue
            x, y, radius = pos
            cls._draw_one_pie(ax, (x, y), radius, ordered, category_to_color)
