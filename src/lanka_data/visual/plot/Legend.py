from lanka_data.visual.plot.Style import Style


class Legend:
    MAX_ITEMS = 10
    MARKER_SIZE = 100
    MAX_LABEL_LEN = 18
    MIN_FONT_SIZE = 6
    LEGEND_KWARGS = {
        "loc": "best",
        "frameon": False,
    }

    @classmethod
    def _fontsize(cls, labels):
        max_len = max((len(label) for label in labels), default=0)
        if max_len <= cls.MAX_LABEL_LEN:
            return Style.FONT_SIZE_METADATA
        scaled = Style.FONT_SIZE_METADATA * cls.MAX_LABEL_LEN / max_len
        return max(cls.MIN_FONT_SIZE, scaled)

    @staticmethod
    def _format_label(value, region=None):
        if isinstance(value, (int, float)):
            text = f"{value:.1%}"
        else:
            text = str(value)
        if region:
            return f"{text} — {region}"
        return text

    @classmethod
    def _trim(cls, value_and_color):
        if len(value_and_color) <= cls.MAX_ITEMS:
            return value_and_color
        n_actual = len(value_and_color)
        n_req = cls.MAX_ITEMS - 1
        trimmed = [
            value_and_color[int(i * n_actual / n_req)] for i in range(n_req)
        ]
        trimmed.append(value_and_color[-1])
        return trimmed

    @classmethod
    def draw(
        cls, value_to_color, legend_ax, title=None, value_to_region=None
    ):
        if not legend_ax.has_data():
            legend_ax.set_axis_off()
        if value_to_color is None:
            return

        value_to_region = value_to_region or {}
        value_and_color = cls._trim(list(value_to_color.items()))
        handles = [
            legend_ax.scatter([], [], color=color, s=cls.MARKER_SIZE)
            for value, color in value_and_color
        ]
        labels = [
            cls._format_label(value, value_to_region.get(value))
            for value, color in value_and_color
        ]

        legend_kwargs = dict(cls.LEGEND_KWARGS)
        legend_kwargs["fontsize"] = cls._fontsize(labels)
        if title:
            legend_kwargs["title"] = title
        legend_ax.legend(
            handles=handles,
            labels=labels,
            **legend_kwargs,
        )
