from matplotlib.ticker import FuncFormatter

from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot_visual.BarChartVisual.BarChartVisual import BarChartVisual

from .HistogramData import HistogramData


class HistogramVisual(BarChartVisual):
    N_BINS = 12
    BAR_COLOR = "#1f77b4"

    @staticmethod
    def _region_totals(subregions):
        totals = []
        for subregion in subregions:
            values = subregion["values"]
            total = subregion.get("total_value")
            if total is None:
                total = sum(values.values())
            if total:
                totals.append(total)
        return totals

    def _draw_bars(self, ax, edges, counts):
        for i, count in enumerate(counts):
            ax.bar(
                edges[i],
                count,
                width=edges[i + 1] - edges[i],
                align="edge",
                color=self.BAR_COLOR,
                edgecolor="white",
                linewidth=0.5,
            )

    def _style_axis(self, ax):
        ax.xaxis.set_major_formatter(FuncFormatter(self._format_millions))
        ax.set_xlabel("Population", color=Style.COLOR_METADATA)
        ax.set_ylabel("Number of regions", color=Style.COLOR_METADATA)
        ax.grid(
            True, axis="y", color=Style.COLOR_GRID, linewidth=0.5, zorder=-1
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def draw(self, dataset, fig):
        subregions = self._build_subregions(dataset.get_data_table())
        totals = self._region_totals(subregions)
        ax = fig.add_subplot(fig.add_gridspec(1, 1)[0])
        edges, counts = HistogramData.bins(totals, self.N_BINS)
        if not counts:
            ax.set_axis_off()
            return
        self._draw_bars(ax, edges, counts)
        self._style_axis(ax)
