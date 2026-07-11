from lanka_data.visual.plot.map.TriangleData.UnitTriangleData import \
    UnitTriangleData
from lanka_data.visual.plot_visual.TriangleMapVisual.TriangleMapVisual import \
    TriangleMapVisual


class UnitTriangleMapVisual(TriangleMapVisual):
    @staticmethod
    def _get_data_list(dataset):
        return dataset.get_data_table()

    @staticmethod
    def _get_layout(data_list):
        return UnitTriangleData.get_triangle_layout(data_list)

    @staticmethod
    def _draw_scale(ax, layout):
        pass
