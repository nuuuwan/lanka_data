from lanka_data.how.How import How
from lanka_data.how.map.MapUtils import MapUtils


class Map(How):
    def get_description(self):
        return "Map"

    def get_inner(self, where, what, when, cmd):
        return MapUtils.draw_map(where, what, when, self, cmd)
