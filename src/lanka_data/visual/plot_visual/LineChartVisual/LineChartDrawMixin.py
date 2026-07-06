from matplotlib.ticker import FuncFormatter

from lanka_data.visual.plot.Legend import Legend
from lanka_data.visual.plot.Style import Style


class LineChartDrawMixin:
    LINE_WIDTH = 2.5
    MARKER_SIZE = 7
    AREA_ALPHA = 0.12

    @staticmethod
    def _format_millions(value, _):
        av = abs(value)
        if av >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if av >= 1_000:
            return f"{value / 1_000:.0f}K"
        return f"{value:.0f}"

    @classmethod
    def _draw_series(cls, ax, x_values, series, selected, category_to_color):
        for cat in selected:
            y_values = series[cat]
            color = category_to_color.get(cat)
            ax.fill_between(
                x_values, y_values, color=color, alpha=cls.AREA_ALPHA
            )
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=cls.LINE_WIDTH,
                marker="o",
                markersize=cls.MARKER_SIZE,
                label=cat,
            )

    def _style_axis(self, ax, x_values, year_labels):
        ax.set_xticks(x_values)
        ax.set_xticklabels(
            year_labels,
            fontsize=Style.FONT_SIZE_METADATA,
            color=Style.COLOR_METADATA,
        )
        ax.margins(x=0.05)
        ax.set_ylim(bottom=0)
        ax.grid(
            True, axis="y", color=Style.COLOR_GRID, linewidth=0.5, zorder=-1
        )
        ax.yaxis.set_major_formatter(FuncFormatter(self._format_millions))
        ax.set_ylabel("Population", color=Style.COLOR_METADATA)
        ax.tick_params(
            axis="y",
            labelsize=Style.FONT_SIZE_METADATA,
            labelcolor=Style.COLOR_METADATA,
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    @staticmethod
    def _draw_legend(ax, selected, category_to_color):
        items = {
            cat: category_to_color[cat]
            for cat in selected
            if cat in category_to_color
        }
        if items:
            Legend.draw(items, ax)
