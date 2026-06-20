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
    @staticmethod
    def from_commmand_and_datasets(command, datasets):
        how_cmd = command.how_cmd
        how_without_params = how_cmd.split(":")[0]

        if how_without_params == "JSON":
            return JSONVisual(
                command=command,
                datasets=datasets,
                how_cmd=command.how_cmd,
            )

        if (
            how_without_params == "Map"
            or how_without_params == "Cartogram"
            or how_without_params == "Basic"
        ):
            return MapVisual(
                command=command,
                datasets=datasets,
                how_cmd=command.how_cmd,
            )

        if how_without_params == "BarChart":
            return BarChartVisual(
                command=command,
                datasets=datasets,
                how_cmd=command.how_cmd,
            )

        if how_without_params == "PieChart":
            return PieChartVisual(
                command=command,
                datasets=datasets,
                how_cmd=command.how_cmd,
            )

        if how_without_params == "BumpChart":
            return BumpChartVisual(
                command=command,
                datasets=datasets,
                how_cmd=command.how_cmd,
            )

        raise ValueError(f"Unknown how_cmd: {command.how_cmd}")
