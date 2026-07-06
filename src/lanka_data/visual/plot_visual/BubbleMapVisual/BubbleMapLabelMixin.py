from lanka_data.visual.plot.LabelTruncator import LabelTruncator
from utils_future import ColorUtils, timer


class BubbleMapLabelMixin:
    IS_LIGHT_COLOR = getattr(ColorUtils, "_is_light_color")

    @classmethod
    def _default_fill_color(cls, color):
        if color and not cls.IS_LIGHT_COLOR(color):
            return "#fff"
        return "#000"

    @staticmethod
    def _bubble_positions(layout):
        return {region_id: (x, y) for region_id, x, y, _ in layout["bubbles"]}

    @classmethod
    def _label(cls, name, region_count):
        return LabelTruncator.get_label(
            name,
            region_count,
            LabelTruncator.HEX_MAX_REGIONS_FULL_LABEL,
            LabelTruncator.HEX_MAX_REGIONS_TRUNCATE_MID,
            LabelTruncator.HEX_MAX_REGIONS_TRUNCATE_SMALL,
        )

    @classmethod
    def _text_color(cls, region_color_map, region_id):
        region_color = region_color_map.get(region_id)
        color = region_color or cls._default_fill_color(region_color)
        return "#666" if cls.IS_LIGHT_COLOR(color) else "#eee"

    @classmethod
    @timer
    def _draw_labels(
        cls, ax, layout, region_to_name, region_color_map, region_count
    ):
        for region_id, (cx, cy) in cls._bubble_positions(layout).items():
            name = region_to_name.get(region_id, str(region_id))
            label = cls._label(name, region_count)
            if label is None:
                continue
            words = label.split()
            longest_word = max(words, key=len)
            ax.annotate(
                "\n".join(words),
                xy=(cx, cy),
                ha="center",
                va="center",
                fontsize=60 / len(longest_word),
                color=cls._text_color(region_color_map, region_id),
            )
