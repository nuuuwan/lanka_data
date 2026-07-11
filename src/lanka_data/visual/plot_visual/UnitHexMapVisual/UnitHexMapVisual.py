from lanka_data.visual.plot.map.HexData.UnitHexData import UnitHexData
from lanka_data.visual.plot_visual.HexMapVisual.HexMapVisual import \
    HexMapVisual


class UnitHexMapVisual(HexMapVisual):
    @classmethod
    def get_description(cls):
        return (
            "Renders data as a unit hexagonal map with exactly one "
            "hexagon per region"
        )

    @staticmethod
    def _get_data_list(dataset):
        return dataset.get_data_table()

    @staticmethod
    def _get_layout(data_list):
        return UnitHexData.get_hex_layout(data_list)

    @staticmethod
    def _draw_scale(ax, layout):
        pass
