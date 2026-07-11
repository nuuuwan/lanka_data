from dataclasses import dataclass

from lanka_data.visual.data_export.ChartSpecVisual import ChartSpecVisual
from lanka_data.visual.data_export.CSVVisual import CSVVisual
from lanka_data.visual.data_export.GeoJSONVisual import GeoJSONVisual
from lanka_data.visual.data_export.ParquetVisual import ParquetVisual
from lanka_data.visual.data_export.TSVVisual import TSVVisual
from lanka_data.visual.JSONVisual import JSONVisual
from lanka_data.visual.plot_visual.BarChartVisual.BarChartVisual import \
    BarChartVisual
from lanka_data.visual.plot_visual.BivariateMapVisual.BivariateMapVisual import \
    BivariateMapVisual
from lanka_data.visual.plot_visual.BivariateMapVisual.QuadrantChartVisual import \
    QuadrantChartVisual
from lanka_data.visual.plot_visual.BubbleMapVisual.BubbleMapVisual import \
    BubbleMapVisual
from lanka_data.visual.plot_visual.BumpChartVisual.BumpChartVisual import \
    BumpChartVisual
from lanka_data.visual.plot_visual.HexMapVisual.HexMapVisual import \
    HexMapVisual
from lanka_data.visual.plot_visual.HistogramVisual.HistogramVisual import \
    HistogramVisual
from lanka_data.visual.plot_visual.LineChartVisual.LineChartVisual import \
    LineChartVisual
from lanka_data.visual.plot_visual.MapVisual import MapVisual
from lanka_data.visual.plot_visual.PieChartVisual.PieChartVisual import \
    PieChartVisual
from lanka_data.visual.plot_visual.ScatterPlotVisual.ScatterPlotVisual import \
    ScatterPlotVisual
from lanka_data.visual.plot_visual.SquareMapVisual.SquareMapVisual import \
    SquareMapVisual
from lanka_data.visual.plot_visual.StackedBarChartVisual.StackedBarChartVisual import \
    StackedBarChartVisual
from lanka_data.visual.plot_visual.TreeMapVisual.TreeMapVisual import \
    TreeMapVisual
from lanka_data.visual.plot_visual.TriangleMapVisual.TriangleMapVisual import \
    TriangleMapVisual
from lanka_data.visual.plot_visual.UnitHexMapVisual.UnitHexMapVisual import \
    UnitHexMapVisual
from lanka_data.visual.plot_visual.UnitSquareMapVisual.UnitSquareMapVisual import \
    UnitSquareMapVisual
from lanka_data.visual.plot_visual.UnitTriangleMapVisual.UnitTriangleMapVisual import \
    UnitTriangleMapVisual
from utils_future import Log

log = Log("VisualFactory")


@dataclass
class VisualFactory:
    CLS_LIST = {
        BarChartVisual,
        BivariateMapVisual,
        BubbleMapVisual,
        BumpChartVisual,
        ChartSpecVisual,
        CSVVisual,
        GeoJSONVisual,
        HexMapVisual,
        HistogramVisual,
        JSONVisual,
        LineChartVisual,
        MapVisual,
        MapVisual,
        MapVisual,
        ParquetVisual,
        PieChartVisual,
        QuadrantChartVisual,
        ScatterPlotVisual,
        SquareMapVisual,
        StackedBarChartVisual,
        TreeMapVisual,
        TriangleMapVisual,
        TSVVisual,
        UnitHexMapVisual,
        UnitSquareMapVisual,
        UnitTriangleMapVisual,
    }

    @staticmethod
    def from_command_and_datasets(command, datasets):
        how_cmd = command.how_cmd
        visual_cls = VisualFactory._VISUAL_CLS.get(command.how.base)
        if visual_cls is None:
            raise ValueError(f"Unknown how_cmd: {command.how_cmd}")
        return visual_cls(command=command, datasets=datasets, how_cmd=how_cmd)

    @staticmethod
    def from_commmand_and_datasets(command, datasets):
        return VisualFactory.from_command_and_datasets(command, datasets)
