from lanka_data.api.how.How import How
from lanka_data.api.how.map.PlotUtils import PlotUtils


class Map(How):

    def get_inner(self, command):
        return PlotUtils.draw_plot(command, is_cartogram=False)
