from lanka_data.dataset.custom.Census2001Dataset import Census2001Dataset
from lanka_data.dataset.custom.Census2012Dataset import Census2012Dataset
from lanka_data.dataset.custom.Census2024Dataset import Census2024Dataset
from lanka_data.dataset.custom.ElectionDataset import ElectionDataset
from lanka_data.dataset.custom.ElectionSummaryDataset import \
    ElectionSummaryDataset
from lanka_data.dataset.DiffDataset import DiffDataset
from lanka_data.dataset.EmptyDataset import EmptyDataset
from lanka_data.region.Regions import Regions
from utils_future import Log
from utils_future.timer import timer

log = Log("DatasetFactory")


class DatasetFactory:

    @timer
    @staticmethod
    def get_region_data_list(command):
        return Regions.from_command(command).raw_region_data_list

    @staticmethod
    def _try_census_dataset(command, region_data_list):
        census_map = {
            "2024": Census2024Dataset,
            "2012": Census2012Dataset,
            "2001": Census2001Dataset,
        }
        cls = census_map.get(command.when_cmd)
        if cls and command.what_cmd in cls.get_labels():
            return cls.from_label_and_region_data_list(
                command.what_cmd, region_data_list
            )
        return None

    @staticmethod
    def _try_election_dataset(command, region_data_list):
        if command.what_cmd in ElectionDataset.get_labels():
            return ElectionDataset.from_label_and_region_data_list_and_year(
                command.what_cmd, region_data_list, command.when_cmd
            )
        base_label = command.what_cmd.replace("Summary", "")
        if base_label in ElectionDataset.get_labels():
            return ElectionSummaryDataset.from_label_and_region_data_list_and_year(
                base_label,
                region_data_list,
                command.when_cmd,
            )
        return None

    @staticmethod
    def from_command(command):
        region_data_list = DatasetFactory.get_region_data_list(command)

        if command.what_cmd == "Empty":
            return EmptyDataset(region_data_list)

        result = DatasetFactory._try_census_dataset(command, region_data_list)
        if result is not None:
            return result

        result = DatasetFactory._try_election_dataset(
            command, region_data_list
        )
        if result is not None:
            return result

        raise ValueError(f"Dataset unknown for: {command}")

    @staticmethod
    def list_from_command(command):
        if "-" in command.when_cmd:
            when_cmd_parts = command.when_cmd.split("-")
            if len(when_cmd_parts) != 2:
                raise ValueError(f"Invalid format: {command}")

            when_cmd_start, when_cmd_end = when_cmd_parts
            command_start = command.copy(when_cmd=when_cmd_start)
            command_end = command.copy(when_cmd=when_cmd_end)

            dataset_start = DatasetFactory.from_command(command_start)
            dataset_end = DatasetFactory.from_command(command_end)
            diff_dataset = DiffDataset(dataset_start, dataset_end)

            return [dataset_start, dataset_end, diff_dataset]

        return [DatasetFactory.from_command(command)]
