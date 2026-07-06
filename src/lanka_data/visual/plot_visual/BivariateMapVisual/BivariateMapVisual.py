from matplotlib.patches import Rectangle

from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot_visual.BivariateMapVisual.\
    BivariateMapBaseVisual import BivariateMapBaseVisual


class BivariateMapVisual(BivariateMapBaseVisual):
    N_BINS = 3
    LEGEND_FONTSIZE = 8

    def _draw_legend_cells(self, ax):
        palette = self.palette
        for y_bin in range(self.N_BINS):
            for x_bin in range(self.N_BINS):
                ax.add_patch(
                    Rectangle(
                        (x_bin, y_bin),
                        1,
                        1,
                        facecolor=palette.color(x_bin, y_bin),
                        edgecolor="white",
                    )
                )

    def _style_legend(self, ax, first, last):
        ax.set_xlim(0, self.N_BINS)
        ax.set_ylim(0, self.N_BINS)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(
            f"{first} share \u2192",
            fontsize=self.LEGEND_FONTSIZE,
            color=Style.COLOR_METADATA,
        )
        ax.set_ylabel(
            f"{last} share \u2192",
            fontsize=self.LEGEND_FONTSIZE,
            color=Style.COLOR_METADATA,
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def _draw_legend(self, ax):
        first, last = self.measure_labels
        self._draw_legend_cells(ax)
        self._style_legend(ax, first, last)

    def draw(self, dataset, fig):
        points = self._classified_points(dataset)
        gdf_region = self._get_gdf_region(
            dataset, self._region_color_map(points)
        )
        gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
        self._draw_map(fig.add_subplot(gs[0]), gdf_region)
        self._draw_legend(fig.add_subplot(gs[1]))
