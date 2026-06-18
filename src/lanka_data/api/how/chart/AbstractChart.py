from lanka_data.api.how.How import How
from lanka_data.api.how.plot import ChartSubFigure, Plot


class AbstractChart(How):
    CHART_TYPE = "Chart"

    def draw_axis(self, ax, chart_data):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_inner(self, command):
        return Plot.draw_plot(
            command,
            is_cartogram=False,
            renderer_class=ChartSubFigure,
        )
