from lanka_data.visual.plot.map.HexData.UnitHexData import UnitHexData
from lanka_data.visual.plot_visual.HexMapVisual.HexMapVisual import (
    HexMapVisual,
)


class UnitHexMapVisual(HexMapVisual):
    @staticmethod
    def _get_data_list(dataset):
        return dataset.get_data_table()

    @staticmethod
    def _get_layout(data_list):
        return UnitHexData.get_hex_layout(data_list)

    @classmethod
    def _draw_scale(cls, ax, layout):
        return
