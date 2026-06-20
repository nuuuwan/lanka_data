from dataclasses import dataclass

from lanka_data.visual.JSONVisual import JSONVisual
from lanka_data.visual.MapVisual import MapVisual
from utils_future import Log

log = Log("VisualFactory")


@dataclass
class VisualFactory:
    @staticmethod
    def from_commmand_and_datasets(command, datasets):
        if command.how_cmd == "JSON":
            return JSONVisual(
                command=command,
                datasets=datasets,
                how_cmd=command.how_cmd,
            )

        if command.how_cmd == "Map" or command.how_cmd == "Cartogram":
            return MapVisual(
                command=command,
                datasets=datasets,
                how_cmd=command.how_cmd,
            )

        raise ValueError(f"Unknown how_cmd: {command.how_cmd}")
