from dataclasses import dataclass

from lanka_data.visual.JSONVisual import JSONVisual
from lanka_data.visual.plot_visual.BarChartVisual import BarChartVisual
from lanka_data.visual.plot_visual.BumpChartVisual import BumpChartVisual
from lanka_data.visual.plot_visual.MapVisual import MapVisual
from lanka_data.visual.plot_visual.PieChartVisual import PieChartVisual
from utils_future import Log

log = Log("VisualFactory")


@dataclass
class VisualFactory:
    _VISUAL_CLS = {
        "JSON": JSONVisual,
        "Map": MapVisual,
        "Cartogram": MapVisual,
        "None": MapVisual,
        "BarChart": BarChartVisual,
        "PieChart": PieChartVisual,
        "BumpChart": BumpChartVisual,
    }

    @staticmethod
    def from_commmand_and_datasets(command, datasets):
        how_cmd = command.how_cmd
        how_without_params = how_cmd.split(":")[0]
        visual_cls = VisualFactory._VISUAL_CLS.get(how_without_params)
        if visual_cls is None:
            raise ValueError(f"Unknown how_cmd: {command.how_cmd}")
        return visual_cls(command=command, datasets=datasets, how_cmd=how_cmd)
