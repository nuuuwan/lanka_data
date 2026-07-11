from lanka_data.visual.plot.Legend import Legend
from lanka_data.visual.plot.map.BubbleData.BubbleData import BubbleData
from lanka_data.visual.plot.map.GeoData.GeoData import GeoData
from lanka_data.visual.plot_visual.BarChartVisual.BarChartVisual import \
    BarChartVisual

from .PieChartMapDrawMixin import PieChartMapDrawMixin
from .PieChartMapLabelMixin import PieChartMapLabelMixin


class PieChartVisual(
    PieChartMapDrawMixin, PieChartMapLabelMixin, BarChartVisual
):
    @staticmethod
    def _order_positive_values_with_top_first(values_map):
        items = [(k, v) for k, v in values_map.items() if v > 0]
        if not items:
            return []
        top_label, _ = max(items, key=lambda x: x[1])
        return [(k, v) for k, v in items if k == top_label] + [
            (k, v) for k, v in items if k != top_label
        ]

    def _draw_change_chart(
        self, ax, subregions, category_labels, category_to_color
    ):
        subregions = self._sort_subregions(subregions)
        if not category_labels:
            ax.set_axis_off()
            return
        x_values = list(range(len(subregions)))
        y_min, y_max = self._draw_stacked_bars(
            ax, subregions, x_values, category_labels, category_to_color
        )
        self._style_axis(ax, subregions, y_min, y_max, "Population")
        self._add_bar_labels(ax, subregions)
        self._draw_category_legend(ax, category_labels, category_to_color)

    def _draw_map_chart(
        self, fig, dataset, subregions, category_labels, category_to_color
    ):
        data_list = dataset.get_data_table()
        gdf_region = GeoData.get_geopandas_dataframe(data_list, False).copy()
        positions = self._positions(BubbleData.get_bubble_layout(data_list))
        gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        legend_ax = fig.add_subplot(gs[1])
        self._draw_background(ax, gdf_region)
        self._draw_map_pies(ax, subregions, positions, category_to_color)
        self._draw_pie_labels(ax, subregions, positions, category_to_color)
        items = {
            lbl: category_to_color[lbl]
            for lbl in category_labels
            if lbl in category_to_color
        }
        Legend.draw(items, legend_ax)

    def draw(self, dataset, fig):
        subregions = self._build_subregions(dataset.get_data_table())
        category_labels = self._build_category_labels(subregions)
        category_to_color = self._build_category_to_color(
            dataset, category_labels
        )
        if not subregions:
            fig.add_subplot(fig.add_gridspec(1, 1)[0]).set_axis_off()
            return
        if self._is_change_chart(subregions):
            ax = fig.add_subplot(fig.add_gridspec(1, 1)[0])
            self._draw_change_chart(
                ax, subregions, category_labels, category_to_color
            )
            return
        self._draw_map_chart(
            fig, dataset, subregions, category_labels, category_to_color
        )
