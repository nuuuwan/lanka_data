from lanka_data.visual.plot_visual.PlotVisual import PlotVisual

from .LineChartDataMixin import LineChartDataMixin
from .LineChartDrawMixin import LineChartDrawMixin


class LineChartVisual(LineChartDataMixin, LineChartDrawMixin, PlotVisual):
    @classmethod
    def get_description(cls):
        return (
            "Renders data as a line chart with categories on x-axis and "
            "values as lines over time or categories"
        )

    def draw(self, dataset, fig):
        data_table = dataset.get_data_table()
        subregions = self._build_subregions(data_table)
        category_labels = self._build_category_labels(subregions)
        category_to_color = self._build_category_to_color(
            dataset, category_labels
        )
        year_labels = self._get_year_labels(dataset, self.command)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        if not category_labels or len(year_labels) < 2:
            ax.set_axis_off()
            return
        series = self._aggregate_series(
            data_table, year_labels, category_labels
        )
        selected = self._select_series(series, category_labels)
        if not selected:
            ax.set_axis_off()
            return
        x_values = list(range(len(year_labels)))
        self._draw_series(ax, x_values, series, selected, category_to_color)
        self._style_axis(ax, x_values, year_labels)
        self._draw_legend(ax, selected, category_to_color)
