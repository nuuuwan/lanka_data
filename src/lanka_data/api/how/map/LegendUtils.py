import numpy as np

from lanka_data.api.how.map.ColorUtils import ColorUtils


def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


class LegendUtils:
    MAX_LEGEND_ITEMS = 7
    LEGEND_2D_N_COLS = 3

    @staticmethod
    def _format_legend_label(value):
        if isinstance(value, (int, float)):
            return f"{value:.1%}"
        return str(value)

    # flake8: noqa: CFQ002
    @staticmethod
    def _build_legend_image(
        value_to_color,
        categories,
        pct_levels,
        pct_min,
        pct_max,
        cat_pct_ranges,
        col_width,
    ):
        n_rows, n_cols = len(categories), len(pct_levels)
        pct_span = pct_max - pct_min if pct_max > pct_min else 1.0
        img = np.zeros((n_rows, n_cols, 4))
        for row_i, cat in enumerate(reversed(categories)):
            r, g, b = value_to_color[cat][:3]
            cat_lo, cat_hi = cat_pct_ranges.get(cat, (pct_min, pct_max))
            for col_j, pct in enumerate(pct_levels):
                if (
                    pct < cat_lo - col_width / 2
                    or pct > cat_hi + col_width / 2
                ):
                    img[row_i, col_j] = [0, 0, 0, 0]
                else:
                    normalised = (pct - pct_min) / pct_span
                    img[row_i, col_j] = [
                        r,
                        g,
                        b,
                        ColorUtils.MIN_ALPHA
                        + normalised * (ColorUtils.ALPHA_SPAN),
                    ]
        return img

    @staticmethod
    def _draw_legend(value_to_color, ax, legend_ax):
        if value_to_color is None:
            legend_ax.set_visible(False)
            return

        legend_ax.set_visible(False)

        value_and_color = list(
            sorted(
                value_to_color.items(),
                key=lambda item: (
                    item[0] if not is_float(item[0]) else float(item[0])
                ),
                reverse=True,
            )
        )

        trimmed = value_and_color
        if len(value_and_color) > LegendUtils.MAX_LEGEND_ITEMS:
            n_actual = len(value_and_color)
            n_req = LegendUtils.MAX_LEGEND_ITEMS - 1
            trimmed = [
                value_and_color[int(i * n_actual / n_req)]
                for i in range(n_req)
            ]
            trimmed.append(value_and_color[-1])

        for value, color in trimmed:
            ax.scatter(
                [],
                [],
                color=color,
                label=LegendUtils._format_legend_label(value),
            )
        ax.legend(fontsize=12)
