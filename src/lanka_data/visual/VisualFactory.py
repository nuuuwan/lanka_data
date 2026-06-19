from dataclasses import dataclass

from lanka_data.visual import Visual
from utils_future import Log

log = Log("VisualFactory")


@dataclass
class VisualFactory:
    @staticmethod
    def from_commmand_and_datasets(command, datasets):
        return Visual(datasets=datasets, how_cmd=command.how_cmd)
