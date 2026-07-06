from utils_future import ColorUtils


class PieChartMapLabelMixin:
    IS_LIGHT_COLOR = getattr(ColorUtils, "_is_light_color")
    MIN_FONT = 3.0
    MAX_FONT = 12.0

    @classmethod
    def _top_field(cls, subregion):
        ordered = cls._order_positive_values_with_top_first(
            subregion["values"]
        )
        if not ordered:
            return None
        total = sum(v for _, v in ordered) or 1
        label, value = ordered[0]
        return label, value / total

    @classmethod
    def _label_color(cls, label, category_to_color):
        color = category_to_color.get(label)
        if color and cls.IS_LIGHT_COLOR(color):
            return "#333"
        return "#eee"

    @staticmethod
    def _radius_px(ax, x, y, radius):
        transform = ax.transData.transform
        return abs(transform((x + radius, y))[0] - transform((x, y))[0])

    @classmethod
    def _fontsize(cls, ax, x, y, radius, n_chars):
        pt = cls._radius_px(ax, x, y, radius) * 72 / ax.get_figure().dpi
        return max(
            cls.MIN_FONT,
            min(cls.MAX_FONT, pt / max(n_chars, 4) * 1.6),
        )

    @classmethod
    def _draw_pie_labels(cls, ax, subregions, positions, category_to_color):
        for subregion in subregions:
            pos = positions.get(subregion["region_id"])
            top = cls._top_field(subregion)
            if pos is None or top is None:
                continue
            label, pct = top
            x, y, radius = pos

            for text, offset, p_font_size in [
                (f"{label}", 0.2, 0.5),
                (f"{pct:.1%}", 0.5, 1.5),
                (subregion["region_name"], 0.8, 0.5),
            ]:
                ax.annotate(
                    text,
                    xy=(x, y + radius * offset),
                    ha="center",
                    va="center",
                    fontsize=cls._fontsize(ax, x, y, radius, len(label))
                    * p_font_size,
                    color=cls._label_color(label, category_to_color),
                )
