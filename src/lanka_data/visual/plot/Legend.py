class Legend:
    MAX_ITEMS = 10
    MARKER_SIZE = 100
    LEGEND_KWARGS = {
        "fontsize": 12,
        "loc": "best",
    }

    @staticmethod
    def _format_label(value):
        if isinstance(value, (int, float)):
            return f"{value:.1%}"
        return str(value)

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
    def draw(cls, value_to_color, legend_ax, title=None):
        if not legend_ax.has_data():
            legend_ax.set_axis_off()
        if value_to_color is None:
            return

        value_and_color = cls._trim(list(value_to_color.items()))
        handles = [
            legend_ax.scatter([], [], color=color, s=cls.MARKER_SIZE)
            for value, color in value_and_color
        ]
        labels = [cls._format_label(value) for value, color in value_and_color]

        legend_kwargs = dict(cls.LEGEND_KWARGS)
        if title:
            legend_kwargs["title"] = title
        legend_ax.legend(
            handles=handles,
            labels=labels,
            **legend_kwargs,
        )
