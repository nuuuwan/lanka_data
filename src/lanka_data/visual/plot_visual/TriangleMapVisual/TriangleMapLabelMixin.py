from lanka_data.visual.plot.Label import Label
from lanka_data.visual.plot.LabelTruncator import LabelTruncator
from lanka_data.visual.plot_visual.TriangleMapVisual.TriangleMapLabelGeoMixin import (
    TriangleMapLabelGeoMixin,
)
from utils_future import ColorUtils, timer


class TriangleMapLabelMixin(TriangleMapLabelGeoMixin):
    IS_LIGHT_COLOR = getattr(ColorUtils, "_is_light_color")
    FIT_FONTSIZE = getattr(Label, "_fit_fontsize")
    TRUNCATED_FONT_SIZE = 8
    LABEL_BOX_FACTOR = 0.6

    @classmethod
    def _default_fill_color(cls, color):
        if color and not cls.IS_LIGHT_COLOR(color):
            return "#fff"
        return "#000"

    @classmethod
    def _text_color(cls, region_color):
        color = region_color or cls._default_fill_color(region_color)
        return "#666" if cls.IS_LIGHT_COLOR(color) else "#eee"

    @staticmethod
    def _is_truncated(region_count):
        return (
            LabelTruncator.get_truncate_length(region_count)
            != LabelTruncator.TRUNCATE_LEN_FULL
        )

    @staticmethod
    def _label(name, region_count):
        return LabelTruncator.get_label(name, region_count)

    @classmethod
    def _annotate(cls, ax, label, position, fontsize, color):
        ax.annotate(
            label,
            xy=position,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
        )

    @classmethod
    def _draw_truncated_labels(
        cls, ax, layout, region_to_name, region_color_map, region_count
    ):
        for region_id, position in cls._region_positions(
            layout, True
        ).items():
            name = region_to_name.get(region_id, str(region_id))
            label = cls._label(name, region_count)
            if label is None:
                continue
            cls._annotate(
                ax,
                label,
                position,
                cls.TRUNCATED_FONT_SIZE,
                cls._text_color(region_color_map.get(region_id)),
            )

    @classmethod
    def _draw_full_labels(
        cls, ax, layout, region_to_name, region_color_map, region_count
    ):
        fig = ax.get_figure()
        size = layout["size"]
        for region_id, points in cls._region_triangles(layout).items():
            name = region_to_name.get(region_id, str(region_id))
            label = cls._label(name, region_count)
            if label is None:
                continue
            cx, cy = cls._region_centroid(points)
            width, height = cls._region_extent(points)
            box_w = (width + size) * cls.LABEL_BOX_FACTOR
            box_h = (height + size) * cls.LABEL_BOX_FACTOR
            fontsize = cls.FIT_FONTSIZE(label, box_w, box_h, ax, fig)
            cls._annotate(
                ax,
                label,
                (cx, cy),
                fontsize,
                cls._text_color(region_color_map.get(region_id)),
            )

    @classmethod
    @timer
    def _draw_labels(
        cls, ax, layout, region_to_name, region_color_map, region_count
    ):
        draw = (
            cls._draw_truncated_labels
            if cls._is_truncated(region_count)
            else cls._draw_full_labels
        )
        draw(ax, layout, region_to_name, region_color_map, region_count)
