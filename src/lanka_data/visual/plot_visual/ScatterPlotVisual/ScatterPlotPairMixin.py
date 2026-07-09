from matplotlib.ticker import PercentFormatter

from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot_visual.BivariateMapVisual import BivariateData

from .ScatterPlotStats import ScatterPlotStats


class ScatterPlotPairMixin:
    FIT_COLOR = "#d1495b"

    @property
    def categories(self):
        return self.command.how.categories

    def _measure_labels(self):
        whats = self.command.what.whats
        first = whats[0]
        last = whats[-1] if len(whats) > 1 else whats[0]
        first_category, last_category = BivariateData._pair(self.categories)
        if first_category:
            first = f"{first_category} ({first})"
        if last_category:
            last = f"{last_category} ({last})"
        return first, last

    def _draw_pair_points(self, ax, points):
        for point in points:
            ax.scatter(
                point["x"],
                point["y"],
                s=self.MARKER_SIZE,
                color=Style.COLOR_AXIS,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.annotate(
                point["region_name"],
                (point["x"], point["y"]),
                fontsize=self.LABEL_FONTSIZE,
                color=Style.COLOR_METADATA,
                xytext=(3, 3),
                textcoords="offset points",
            )

    def _draw_fit(self, ax, points):
        xs = [point["x"] for point in points]
        ys = [point["y"] for point in points]
        fit = ScatterPlotStats.fit(xs, ys)
        if fit is None:
            return
        line_x, line_y = ScatterPlotStats.line_endpoints(fit, xs)
        ax.plot(line_x, line_y, color=self.FIT_COLOR, linewidth=1.5, zorder=2)
        ax.text(
            0.03,
            0.97,
            ScatterPlotStats.text(fit),
            transform=ax.transAxes,
            fontsize=self.LABEL_FONTSIZE + 1,
            color=Style.COLOR_PANEL,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor=Style.COLOR_BORDER,
                alpha=0.9,
            ),
        )

    def _style_pair_axis(self, ax):
        first, last = self._measure_labels()
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_xlabel(f"{first} share", color=Style.COLOR_METADATA)
        ax.set_ylabel(f"{last} share", color=Style.COLOR_METADATA)
        ax.grid(True, color=Style.COLOR_GRID, linewidth=0.5, zorder=-1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def _draw_pair(self, dataset, fig):
        points = BivariateData.points(
            dataset.get_data_table(), self.categories
        )
        ax = fig.add_subplot(fig.add_gridspec(1, 1)[0])
        if not points:
            ax.set_axis_off()
            return
        self._draw_pair_points(ax, points)
        self._draw_fit(ax, points)
        self._style_pair_axis(ax)
