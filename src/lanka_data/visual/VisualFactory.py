from dataclasses import dataclass

from lanka_data.visual.JSONVisual import JSONVisual
from utils_future import Log

log = Log("VisualFactory")


@dataclass
class VisualFactory:
    @staticmethod
    def from_commmand_and_datasets(command, datasets):
        return JSONVisual(datasets=datasets, how_cmd=command.how_cmd)
