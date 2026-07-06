from lanka_data.visual.plot_visual.BarChartVisual import BarChartVisual

from .TreeMapData import TreeMapData
from .TreeMapDrawMixin import TreeMapDrawMixin


class TreeMapVisual(TreeMapDrawMixin, BarChartVisual):
    @staticmethod
    def _category_totals(subregions):
        totals = {}
        for subregion in subregions:
            for category, value in subregion["values"].items():
                if value > 0:
                    totals[category] = totals.get(category, 0) + value
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

    def draw(self, dataset, fig):
        subregions = self._build_subregions(dataset.get_data_table())
        category_labels = self._build_category_labels(subregions)
        category_to_color = self._build_category_to_color(
            dataset, category_labels
        )
        totals = self._category_totals(subregions)
        ax = fig.add_subplot(fig.add_gridspec(1, 1)[0])
        if not totals:
            ax.set_axis_off()
            return
        labels = list(totals.keys())
        rects = TreeMapData.layout(
            [totals[label] for label in labels], 0, 0, 1, 1
        )
        self._draw_treemap(ax, rects, labels, totals, category_to_color)
