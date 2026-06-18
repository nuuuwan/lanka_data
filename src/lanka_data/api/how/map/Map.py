from lanka_data.api.how.How import How
from lanka_data.api.how.plot import MapSubFigure, Plot


class Map(How):

    def draw_axis(self, ax, chart_data):
        from lanka_data.api.how.chart.BarChart import BarChart

        BarChart(self.how_label, self.params).draw_axis(ax, chart_data)

    def get_inner(self, command):
        return Plot.draw_plot(
            command,
            is_cartogram=False,
            renderer_class=MapSubFigure,
        )
