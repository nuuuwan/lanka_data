import numpy as np


class LegendUtils:
    MAX_LEGEND_ITEMS = 7

    @staticmethod
    def _format_legend_label(value):
        if isinstance(value, (int, float)):
            return f"{value:.1%}"
        return str(value)

    @staticmethod
    def _build_legend_image(
        value_to_color,
        categories,
        pct_levels,
        pct_min,
        pct_max,
        cat_pct_ranges,
        col_half_width,
    ):
        n_rows, n_cols = len(categories), len(pct_levels)
        pct_span = pct_max - pct_min if pct_max > pct_min else 1.0
        img = np.zeros((n_rows, n_cols, 4))
        for row_i, cat in enumerate(reversed(categories)):
            r, g, b = value_to_color[cat][:3]
            cat_lo, cat_hi = cat_pct_ranges.get(cat, (pct_min, pct_max))
            for col_j, pct in enumerate(pct_levels):
                if (
                    pct < cat_lo - col_half_width
                    or pct > cat_hi + col_half_width
                ):
                    img[row_i, col_j] = [0, 0, 0, 0]
                    continue
                normalised = (pct - pct_min) / pct_span
                img[row_i, col_j] = [r, g, b, 0.5 + normalised * 0.5]
        return img

    @staticmethod
    def _draw_legend_2d(value_to_color, legend_ax):
        pct_range = value_to_color.pop("__pct_range__", (0.0, 1.0))
        cat_pct_ranges = value_to_color.pop("__cat_pct_ranges__", {})
        pct_min, pct_max = pct_range
        categories = sorted(value_to_color.keys(), key=str)
        n_cols = 5
        pct_levels = [
            pct_min + i * (pct_max - pct_min) / 4 for i in range(n_cols)
        ]
        col_half_width = (
            (pct_levels[1] - pct_levels[0]) / 2 if n_cols > 1 else 0
        )
        img = LegendUtils._build_legend_image(
            value_to_color,
            categories,
            pct_levels,
            pct_min,
            pct_max,
            cat_pct_ranges,
            col_half_width,
        )
        n_rows = len(categories)
        grid_h = min(n_rows * 0.07, 0.9)
        inset = legend_ax.inset_axes([0.0, (1.0 - grid_h) / 2, 1.0, grid_h])
        legend_ax.set_axis_off()
        inset.imshow(img, aspect="auto", interpolation="nearest")
        for spine in inset.spines.values():
            spine.set_visible(False)
        inset.set_xticks(range(n_cols))
        inset.set_xticklabels([f"{p:.0%}" for p in pct_levels], fontsize=7)
        inset.xaxis.set_ticks_position("top")
        inset.xaxis.set_label_position("top")
        inset.set_xlabel("% share", fontsize=7, labelpad=3)
        inset.set_yticks(range(n_rows))
        inset.set_yticklabels(
            [str(c) for c in reversed(categories)], fontsize=7
        )
        inset.tick_params(axis="both", length=0, pad=2)
        for x in [i - 0.5 for i in range(n_cols + 1)]:
            inset.axvline(x, color="white", linewidth=0.5)
        for y in [i - 0.5 for i in range(n_rows + 1)]:
            inset.axhline(y, color="white", linewidth=0.5)

    @staticmethod
    def _draw_legend(value_to_color, ax, legend_ax):
        if value_to_color is None:
            legend_ax.set_visible(False)
            return
        colors = list(value_to_color.values())
        if colors and isinstance(colors[0], tuple) and len(colors[0]) == 4:
            LegendUtils._draw_legend_2d(value_to_color, legend_ax)
            return
        legend_ax.set_visible(False)
        value_and_color = sorted(value_to_color.items(), reverse=True)
        if len(value_and_color) > LegendUtils.MAX_LEGEND_ITEMS:
            n_actual = len(value_and_color)
            n_req = LegendUtils.MAX_LEGEND_ITEMS - 1
            trimmed = [
                value_and_color[int(i * n_actual / n_req)]
                for i in range(n_req)
            ]
            trimmed.append(value_and_color[-1])
            value_and_color = trimmed
        for value, color in value_and_color:
            ax.scatter(
                [],
                [],
                color=color,
                label=LegendUtils._format_legend_label(value),
            )
        ax.legend(fontsize=6)
