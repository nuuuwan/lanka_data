from lanka_data.dataset.Dataset import Dataset
from lanka_data.dataset.DiffDataset import DiffDataset
from utils_future import Log

log = Log("DatasetFactory")


class DatasetFactory:

    @staticmethod
    def from_command(command):
        if command.what_cmd == "Religion":
            return Dataset()

        raise ValueError(f"Unknown dataset for: {command}")

    @staticmethod
    def list_from_command(command):
        if "-" in command.when_cmd:
            when_cmd_parts = command.when_cmd.split("-")
            if len(when_cmd_parts) != 2:
                raise ValueError(f"Invalid format: {command}")
            when_cmd_start, when_cmd_end = when_cmd_parts
            dataset_start = command.copy(when_cmd=when_cmd_start)
            dataset_end = command.copy(when_cmd=when_cmd_end)
            diff_dataset = DiffDataset(dataset_start, dataset_end)
            return [dataset_start, dataset_end, diff_dataset]

        return [DatasetFactory.from_command(command)]
