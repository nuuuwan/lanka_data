import math

from lanka_data.visual.plot.Style import Style


class PieChartGridMixin:
    @classmethod
    def _draw_single_pie(cls, ax, grid, idx, subregion, category_to_color):
        n_cols, n_rows = grid
        row, col = divmod(idx, n_cols)
        w, h = 1 / n_cols, 1 / n_rows
        inset = ax.inset_axes(
            [
                col / n_cols + 0.02 * w,
                1 - (row + 1) / n_rows + 0.05 * h,
                w * 0.9,
                h * 0.9,
            ]
        )
        ordered = cls._order_positive_values_with_top_first(
            subregion["values"]
        )
        if not ordered:
            inset.set_axis_off()
            return
        vals = [v for _, v in ordered]
        inset.pie(
            vals,
            colors=[category_to_color[k] for k, _ in ordered],
            wedgeprops={"linewidth": 0.2, "edgecolor": "white"},
        )
        inset.text(
            0.5,
            1.1,
            f"{subregion['region_name']}",
            transform=inset.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            color=Style.COLOR_METADATA,
            clip_on=False,
        )
        inset.axis("equal")

    @classmethod
    def _draw_grid_pies(cls, ax, subregions, category_to_color):
        if not subregions:
            ax.set_axis_off()
            return
        n = len(subregions)
        n_cols = max(1, math.ceil(math.sqrt(n)))
        n_rows = math.ceil(n / n_cols)
        ax.set_axis_off()
        grid = (n_cols, n_rows)
        for idx, subregion in enumerate(subregions):
            cls._draw_single_pie(ax, grid, idx, subregion, category_to_color)
