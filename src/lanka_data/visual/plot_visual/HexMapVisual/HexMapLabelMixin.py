from lanka_data.visual.plot.LabelTruncator import LabelTruncator
from utils_future import ColorUtils, timer


class HexMapLabelMixin:
    IS_LIGHT_COLOR = getattr(ColorUtils, "_is_light_color")

    @classmethod
    def _default_fill_color(cls, color):
        if color and not cls.IS_LIGHT_COLOR(color):
            return "#fff"
        return "#000"

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
    def _region_positions(cls, layout, snap):
        positions = {}
        for region_id, points in cls._region_hexes(layout).items():
            centroid = cls._region_centroid(points)
            positions[region_id] = (
                min(
                    points,
                    key=lambda point: (point[0] - centroid[0]) ** 2
                    + (point[1] - centroid[1]) ** 2,
                )
                if snap
                else centroid
            )
        return positions

    @staticmethod
    def _is_truncated(region_count):
        return (
            LabelTruncator.get_truncate_length(
                region_count,
                LabelTruncator.HEX_MAX_REGIONS_FULL_LABEL,
                LabelTruncator.HEX_MAX_REGIONS_TRUNCATE_MID,
                LabelTruncator.HEX_MAX_REGIONS_TRUNCATE_SMALL,
            )
            != LabelTruncator.TRUNCATE_LEN_FULL
        )

    @classmethod
    @timer
    def _draw_labels(
        cls, ax, layout, region_to_name, region_color_map, region_count
    ):
        positions = cls._region_positions(
            layout, cls._is_truncated(region_count)
        )
        for region_id, (cx, cy) in positions.items():
            name = region_to_name.get(region_id, str(region_id))
            label = LabelTruncator.get_label(
                name,
                region_count,
                LabelTruncator.HEX_MAX_REGIONS_FULL_LABEL,
                LabelTruncator.HEX_MAX_REGIONS_TRUNCATE_MID,
                LabelTruncator.HEX_MAX_REGIONS_TRUNCATE_SMALL,
            )
            if label is None:
                continue
            region_color = region_color_map.get(region_id)
            color = region_color or cls._default_fill_color(region_color)
            text_color = "#666" if cls.IS_LIGHT_COLOR(color) else "#eee"
            ax.annotate(
                label,
                xy=(cx, cy),
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
