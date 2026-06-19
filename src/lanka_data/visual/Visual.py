from dataclasses import dataclass

from lanka_data.dataset.Dataset import Dataset


@dataclass
class Visual:
    datasets: list[Dataset]
    how_cmd: str

    def __str__(self):
        return f"Visual({self.datasets}/{self.how_cmd})"

    @classmethod
    def from_commmand_and_datasets(cls, command, datasets):
        return cls(datasets=datasets, how_cmd=command.how_cmd)
