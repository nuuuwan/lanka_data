from matplotlib.ticker import FuncFormatter, PercentFormatter

from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot_visual.BarChartVisual.BarChartVisual import \
    BarChartVisual

from .ScatterPlotData import ScatterPlotData
from .ScatterPlotPairMixin import ScatterPlotPairMixin


class ScatterPlotVisual(ScatterPlotPairMixin, BarChartVisual):
    MARKER_SIZE = 80
    LABEL_FONTSIZE = 7

    @classmethod
    def get_description(cls):
        return (
            "Renders data as a scatter plot comparing two categories with "
            "fitted correlation line and statistics"
        )

    def _draw_points(self, ax, points, category_to_color):
        for total, share, label, name in points:
            color = category_to_color.get(label, "#999999")
            ax.scatter(
                total,
                share,
                s=self.MARKER_SIZE,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.annotate(
                name,
                (total, share),
                fontsize=self.LABEL_FONTSIZE,
                color=Style.COLOR_METADATA,
                xytext=(3, 3),
                textcoords="offset points",
            )

    def _style_axis(self, ax):
        ax.xaxis.set_major_formatter(FuncFormatter(self._format_millions))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_xlabel("Population", color=Style.COLOR_METADATA)
        ax.set_ylabel("Dominant category share", color=Style.COLOR_METADATA)
        ax.grid(True, color=Style.COLOR_GRID, linewidth=0.5, zorder=-1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def draw(self, dataset, fig):
        if self.command.what.is_combined:
            self._draw_pair(dataset, fig)
            return
        subregions = self._build_subregions(dataset.get_data_table())
        category_labels = self._build_category_labels(subregions)
        category_to_color = self._build_category_to_color(
            dataset, category_labels
        )
        points = ScatterPlotData.points(subregions)
        ax = fig.add_subplot(fig.add_gridspec(1, 1)[0])
        if not points:
            ax.set_axis_off()
            return
        self._draw_points(ax, points, category_to_color)
        self._style_axis(ax)
        self._draw_category_legend(ax, category_labels, category_to_color)
