def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def parse_float(value):
    value_str = (
        str(value)
        .replace(",", "")
        .replace("+", "")
        .replace("%", "")
        .replace("pp", "")
    )

    try:
        return float(value_str)
    except ValueError:
        return None


class LegendUtils:
    MAX_LEGEND_ITEMS = 7

    @staticmethod
    def _format_legend_label(value):
        if isinstance(value, (int, float)):
            return f"{value:.1%}"
        return str(value)

    @staticmethod
    def draw_legend(value_to_color, legend_ax):
        legend_ax.set_axis_off()
        if value_to_color is None:
            return

        value_and_color = list(value_to_color.items())
        trimmed = value_and_color
        if len(value_and_color) > LegendUtils.MAX_LEGEND_ITEMS:
            n_actual = len(value_and_color)
            n_req = LegendUtils.MAX_LEGEND_ITEMS - 1
            trimmed = [
                value_and_color[int(i * n_actual / n_req)]
                for i in range(n_req)
            ]
            trimmed.append(value_and_color[-1])

        handles = [
            legend_ax.scatter([], [], color=color, s=100)
            for value, color in trimmed
        ]
        labels = [
            LegendUtils._format_legend_label(value)
            for value, color in trimmed
        ]
        legend_ax.legend(
            handles=handles,
            labels=labels,
            fontsize=12,
            loc="best",
        )
