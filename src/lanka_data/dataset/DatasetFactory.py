from lanka_data.api.where.Regions import Regions
from lanka_data.dataset.custom.Census2012Dataset import Census2012Dataset
from lanka_data.dataset.custom.Census2024Dataset import Census2024Dataset
from lanka_data.dataset.DiffDataset import DiffDataset
from utils_future import Log

log = Log("DatasetFactory")


class DatasetFactory:

    @staticmethod
    def get_region_ids(command):
        return Regions.from_command(command).region_ids

    @staticmethod
    def from_command(command):
        region_ids = DatasetFactory.get_region_ids(command)
        if (
            command.when_cmd == "2024"
            and command.what_cmd in Census2024Dataset.get_labels()
        ):
            return Census2024Dataset.from_label_and_region_ids(
                command.what_cmd, region_ids
            )

        if (
            command.when_cmd == "2012"
            and command.what_cmd in Census2012Dataset.get_labels()
        ):
            return Census2012Dataset.from_label_and_region_ids(
                command.what_cmd, region_ids
            )

        raise ValueError(f"Dataset unknown for: {command}")

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
