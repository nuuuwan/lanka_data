from lanka_data.api.how.How import How
from lanka_data.api.how.map.PlotUtils import PlotUtils


class Map(How):

    def get_inner(self, what, when, where, cmd):
        return PlotUtils.draw_plot(
            what, when, where, self, cmd, is_cartogram=False
        )
