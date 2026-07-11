from matplotlib.ticker import PercentFormatter

from lanka_data.visual.plot.Style import Style

from .BivariateData import BivariateData
from .BivariateMapBaseVisual import BivariateMapBaseVisual


class QuadrantChartVisual(BivariateMapBaseVisual):
    N_BINS = 2
    MARKER_SIZE = 60
    LABEL_FONTSIZE = 6

    @classmethod
    def get_description(cls):
        return (
            "Renders data as a quadrant chart dividing regions into 4 "
            "quadrants based on two variable values"
        )

    def _draw_points(self, ax, points):
        palette = self.palette
        for point in points:
            ax.scatter(
                point["x"],
                point["y"],
                s=self.MARKER_SIZE,
                color=palette.color(point["x_bin"], point["y_bin"]),
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

    def _draw_crosshairs(self, ax, points):
        x_cut = BivariateData.thresholds([p["x"] for p in points], 2)
        y_cut = BivariateData.thresholds([p["y"] for p in points], 2)
        if x_cut:
            ax.axvline(x_cut[0], color=Style.COLOR_AXIS, linewidth=0.7)
        if y_cut:
            ax.axhline(y_cut[0], color=Style.COLOR_AXIS, linewidth=0.7)

    def _style_scatter(self, ax):
        first, last = self.measure_labels
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_xlabel(f"{first} share", color=Style.COLOR_METADATA)
        ax.set_ylabel(f"{last} share", color=Style.COLOR_METADATA)
        ax.grid(True, color=Style.COLOR_GRID, linewidth=0.5, zorder=-1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def _draw_scatter(self, ax, points):
        if not points:
            ax.set_axis_off()
            return
        self._draw_crosshairs(ax, points)
        self._draw_points(ax, points)
        self._style_scatter(ax)

    def draw(self, dataset, fig):
        points = self._classified_points(dataset)
        self._draw_scatter(fig.add_subplot(1, 1, 1), points)
