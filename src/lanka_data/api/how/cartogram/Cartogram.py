from lanka_data.api.how.map import Map
from lanka_data.api.how.map.PlotUtils import PlotUtils


class Cartogram(Map):

    def get_inner(self, what, when, where, cmd):
        return PlotUtils.draw_plot(
            what, when, where, self, cmd, is_cartogram=True
        )
