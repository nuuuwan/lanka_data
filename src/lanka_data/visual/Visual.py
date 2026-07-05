from abc import ABC, abstractmethod
from dataclasses import dataclass

from lanka_data.api.data.DataSource import DataSource
from lanka_data.api.dataset.Dataset import Dataset
from lanka_data.api.command.Command import Command
from utils_future import Log

log = Log("Visual")


@dataclass
class Visual(ABC):
    command: Command
    datasets: list[Dataset]
    how_cmd: str

    def __str__(self):
        return f"{self.__class__.__name__}({self.datasets} -> {self.how_cmd})"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_command_and_datasets(cls, command, datasets):
        visual = cls(
            command=command,
            datasets=datasets,
            how_cmd=command.how_cmd,
        )
        log.debug(f"Built {visual}")
        return visual

    @classmethod
    def from_commmand_and_datasets(cls, command, datasets):
        return cls.from_command_and_datasets(command, datasets)

    @abstractmethod
    def build(self):
        pass

    def get_sources(self):
        return DataSource.merge_datasource_list_of_lists(
            [dataset.get_sources() for dataset in self.datasets]
        )
