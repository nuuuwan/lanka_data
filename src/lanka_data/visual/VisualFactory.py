from dataclasses import dataclass

from lanka_data.visual.data_export.ChartSpecVisual import ChartSpecVisual
from lanka_data.visual.data_export.CSVVisual import CSVVisual
from lanka_data.visual.data_export.GeoJSONVisual import GeoJSONVisual
from lanka_data.visual.data_export.ParquetVisual import ParquetVisual
from lanka_data.visual.data_export.TSVVisual import TSVVisual
from lanka_data.visual.JSONVisual import JSONVisual
from lanka_data.visual.plot_visual.BarChartVisual import BarChartVisual
from lanka_data.visual.plot_visual.BivariateMapVisual import (
    BivariateMapVisual, QuadrantChartVisual)
from lanka_data.visual.plot_visual.BubbleMapVisual import BubbleMapVisual
from lanka_data.visual.plot_visual.BumpChartVisual import BumpChartVisual
from lanka_data.visual.plot_visual.HexMapVisual import HexMapVisual
from lanka_data.visual.plot_visual.HistogramVisual import HistogramVisual
from lanka_data.visual.plot_visual.LineChartVisual import LineChartVisual
from lanka_data.visual.plot_visual.MapVisual import MapVisual
from lanka_data.visual.plot_visual.PieChartVisual import PieChartVisual
from lanka_data.visual.plot_visual.ScatterPlotVisual import ScatterPlotVisual
from lanka_data.visual.plot_visual.StackedBarChartVisual import \
    StackedBarChartVisual
from lanka_data.visual.plot_visual.TreeMapVisual import TreeMapVisual
from utils_future import Log

log = Log("VisualFactory")


@dataclass
class VisualFactory:
    _VISUAL_CLS = {
        "JSON": JSONVisual,
        "CSV": CSVVisual,
        "TSV": TSVVisual,
        "GeoJSON": GeoJSONVisual,
        "Parquet": ParquetVisual,
        "ChartSpec": ChartSpecVisual,
        "Map": MapVisual,
        "Cartogram": MapVisual,
        "HexMap": HexMapVisual,
        "BubbleMap": BubbleMapVisual,
        "None": MapVisual,
        "BarChart": BarChartVisual,
        "StackedBarChart": StackedBarChartVisual,
        "PieChart": PieChartVisual,
        "BumpChart": BumpChartVisual,
        "TreeMap": TreeMapVisual,
        "Histogram": HistogramVisual,
        "ScatterPlot": ScatterPlotVisual,
        "BivariateMap": BivariateMapVisual,
        "QuadrantChart": QuadrantChartVisual,
        "LineChart": LineChartVisual,
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
