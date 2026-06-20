from abc import ABC, abstractmethod
from dataclasses import dataclass

from lanka_data.dataset.Dataset import Dataset
from utils_future import Log

log = Log("Visual")


@dataclass
class Visual(ABC):
    datasets: list[Dataset]
    how_cmd: str

    def __str__(self):
        return f"{self.__class__.__name__}({self.datasets} -> {self.how_cmd})"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_commmand_and_datasets(cls, command, datasets):
        visual = cls(datasets=datasets, how_cmd=command.how_cmd)
        log.debug(f"Built {visual}")
        return visual

    @abstractmethod
    def build(self):
        pass
