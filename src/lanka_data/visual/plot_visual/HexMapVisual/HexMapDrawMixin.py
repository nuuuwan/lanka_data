from matplotlib.patches import RegularPolygon

from lanka_data.visual.plot.LabelTruncator import LabelTruncator
from utils_future import ColorUtils, timer


class HexMapDrawMixin:
    DEFAULT_EDGE_WIDTH = 0.1
    DEFAULT_FILL_COLOR = "#cccccc"
    IS_LIGHT_COLOR = getattr(ColorUtils, "_is_light_color")

    @classmethod
    def _draw_hexes(cls, ax, layout, region_color_map):
        radius = layout["radius"]
        for region_id, x, y in layout["hexes"]:
            color = region_color_map.get(region_id) or cls.DEFAULT_FILL_COLOR
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

    @staticmethod
    def _region_hexes(layout):
        hexes = {}
        for region_id, x, y in layout["hexes"]:
            hexes.setdefault(region_id, []).append((x, y))
        return hexes

    @classmethod
    def _region_centers(cls, layout):
        centers = {}
        for region_id, points in cls._region_hexes(layout).items():
            point_count = len(points)
            centroid_x = sum(x for x, _ in points) / point_count
            centroid_y = sum(y for _, y in points) / point_count
            centers[region_id] = min(
                points,
                key=lambda point: (point[0] - centroid_x) ** 2
                + (point[1] - centroid_y) ** 2,
            )
        return centers

    @classmethod
    @timer
    def _draw_labels(
        cls, ax, layout, region_to_name, region_color_map, region_count
    ):
        for region_id, (cx, cy) in cls._region_centers(layout).items():
            name = region_to_name.get(region_id, str(region_id))
            label = LabelTruncator.get_label(name, region_count)
            if label is None:
                continue
            color = region_color_map.get(region_id) or cls.DEFAULT_FILL_COLOR
            text_color = "#666" if cls.IS_LIGHT_COLOR(color) else "#eee"
            ax.annotate(
                label,
                xy=(cx, cy),
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
