from lanka_data.api.how.How import How
from lanka_data.api.how.map.MapUtils import MapUtils


class Map(How):

    def get_inner(self, where, what, when, cmd):
        return MapUtils.draw_map(
            where, what, when, self, cmd, is_cartogram=False
        )
