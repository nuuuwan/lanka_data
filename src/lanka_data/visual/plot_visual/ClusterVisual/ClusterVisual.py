import matplotlib.pyplot as plt

from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot_visual.BarChartVisual.BarChartVisual import (
    BarChartVisual,
)

from .ClusterData import ClusterData


class ClusterVisual(BarChartVisual):
    N_CLUSTERS = 5
    CMAP = "viridis"

    @staticmethod
    def _totals_and_names(subregions):
        pairs = []
        for subregion in subregions:
            total = subregion.get("total_value")
            if total is None:
                total = sum(subregion["values"].values())
            if total:
                pairs.append((total, subregion["region_name"]))
        return pairs

    def _draw_points(self, ax, pairs, labels, centers):
        cmap = plt.get_cmap(self.CMAP)
        n_clusters = max(len(centers), 1)
        for (total, name), cluster in zip(pairs, labels):
            color = cmap(cluster / n_clusters)
            ax.scatter(total, cluster, color=color, s=60, zorder=3)
            ax.annotate(
                name,
                (total, cluster),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=Style.FONT_SIZE_METADATA,
                color=Style.COLOR_METADATA,
            )

    def _style_axis(self, ax, centers):
        ax.set_yticks(list(range(len(centers))))
        ax.set_yticklabels(
            [self._format_millions(center, None) for center in centers]
        )
        ax.set_xlabel("Region total", color=Style.COLOR_METADATA)
        ax.set_ylabel("Cluster centre", color=Style.COLOR_METADATA)
        ax.grid(
            True, axis="x", color=Style.COLOR_GRID, linewidth=0.5, zorder=-1
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def draw(self, dataset, fig):
        subregions = self._build_subregions(dataset.get_data_table())
        pairs = self._totals_and_names(subregions)
        ax = fig.add_subplot(fig.add_gridspec(1, 1)[0])
        if not pairs:
            ax.set_axis_off()
            return
        values = [total for total, _ in pairs]
        labels, centers = ClusterData.cluster(values, self.N_CLUSTERS)
        self._draw_points(ax, pairs, labels, centers)
        self._style_axis(ax, centers)
