from lanka_data.visual.plot.map.SquareData.UnitSquareData import UnitSquareData
from lanka_data.visual.plot_visual.SquareMapVisual.SquareMapVisual import \
    SquareMapVisual


class UnitSquareMapVisual(SquareMapVisual):
    @staticmethod
    def _get_data_list(dataset):
        return dataset.get_data_table()

    @staticmethod
    def _get_layout(data_list):
        return UnitSquareData.get_square_layout(data_list)

    @staticmethod
    def _draw_scale(ax, layout):
        pass
