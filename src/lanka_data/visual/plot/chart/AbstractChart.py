from lanka_data.visual.plot import ChartSubFigure, Plot


class AbstractChart:
    CHART_TYPE = "Chart"

    def draw_axis(self, ax, chart_data):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_inner(self, command):
        return Plot.draw(
            command,
            is_cartogram=False,
            renderer_class=ChartSubFigure,
        )
