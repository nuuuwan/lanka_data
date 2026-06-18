from lanka_data.api.how.map import Map
from lanka_data.api.how.plot import MapSubFigure, Plot


class Cartogram(Map):

    def get_inner(self, command):
        return Plot.draw_plot(
            command,
            is_cartogram=True,
            renderer_class=MapSubFigure,
        )
