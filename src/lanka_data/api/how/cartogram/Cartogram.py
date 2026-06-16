from lanka_data.api.how.map import Map
from lanka_data.api.how.map.PlotUtils import PlotUtils


class Cartogram(Map):

    def get_inner(self, where, what, when, cmd):
        return PlotUtils.draw_map(
            where, what, when, self, cmd, is_cartogram=True
        )
