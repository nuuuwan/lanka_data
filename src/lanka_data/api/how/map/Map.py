from lanka_data.api.how.How import How
from lanka_data.api.how.plot import MapSubFigure, Plot


class Map(How):

    def get_inner(self, command):
        return Plot.draw_plot(
            command,
            is_cartogram=False,
            renderer_class=MapSubFigure,
        )
