from lanka_data.how.How import How
from lanka_data.how.MapUtils import MapUtils


class Map(How):
    def get_description(self):
        return "Geographical map visualization"

    def get_inner(self, where, what, when):
        return MapUtils.draw_map(where, what, when, self)
