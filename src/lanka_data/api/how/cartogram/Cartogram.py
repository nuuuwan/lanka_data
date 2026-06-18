from lanka_data.api.how.map import Map
from lanka_data.api.how.plot import Plot


class Cartogram(Map):

    def get_inner(self, command):
        return Plot.draw_plot(command, is_cartogram=True)
