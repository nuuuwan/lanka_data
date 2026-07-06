from lanka_data.api.command_errors.UnknownWhatError import UnknownWhatError
from lanka_data.api.dataset.CorrelationDataset import CorrelationDataset
from lanka_data.api.dataset.DiffDataset import DiffDataset
from lanka_data.datasets.dataset.custom.Census2001Dataset import (
    Census2001Dataset,
)
from lanka_data.datasets.dataset.custom.Census2012Dataset import (
    Census2012Dataset,
)
from lanka_data.datasets.dataset.custom.Census2024Dataset import (
    Census2024Dataset,
)
from lanka_data.datasets.dataset.custom.ElectionDataset import ElectionDataset
from lanka_data.datasets.dataset.custom.ElectionSummaryDataset import (
    ElectionSummaryDataset,
)
from lanka_data.datasets.dataset.EmptyDataset import EmptyDataset
from lanka_data.datasets.region.Regions import Regions
from utils_future import Log
from utils_future.timer import timer

log = Log("DatasetFactory")


class DatasetFactory:
    CENSUS_DATASET_CLASSES = [
        Census2001Dataset,
        Census2012Dataset,
        Census2024Dataset,
    ]
    ELECTION_DATASET_CLASSES = [
        ElectionDataset,
        ElectionSummaryDataset,
    ]

    @timer
    @staticmethod
    def get_region_data_list(command):
        return Regions.from_command(command).raw_region_data_list

    @staticmethod
    def _try_census_dataset(command, region_data_list):
        for cls in DatasetFactory.CENSUS_DATASET_CLASSES:
            if not DatasetFactory._dataset_supports_census(cls, command):
                continue
            return cls.from_label_and_region_data_list(
                command.what_cmd, region_data_list
            )
        return None

    @staticmethod
    def _dataset_supports_census(dataset_cls, command):
        return (
            command.when_cmd in dataset_cls.get_supported_whens()
            and command.what_cmd in dataset_cls.get_labels()
        )

    @staticmethod
    def _try_election_dataset(command, region_data_list):
        if ElectionDataset.supports(command.what_cmd, command.when_cmd):
            return ElectionDataset.from_label_and_region_data_list_and_year(
                command.what_cmd, region_data_list, command.when_cmd
            )
        if ElectionSummaryDataset.supports(
            command.what_cmd, command.when_cmd
        ):
            return ElectionSummaryDataset.from_summary_label_data_and_year(
                command.what_cmd,
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

        raise UnknownWhatError(
            f"Dataset unknown for what: {command.what_cmd}",
            command.what_cmd,
        )

    @staticmethod
    def _list_from_interval(command):
        years = command.when.years
        datasets = [
            DatasetFactory.from_command(
                command.copy(when_cmd=year, how_cmd="")
            )
            for year in years
        ]
        return [DiffDataset(datasets[0], datasets[-1])]

    @staticmethod
    def _list_from_combined(command):
        whats = command.what.whats
        datasets = [
            DatasetFactory.from_command(
                command.copy(what_cmd=what, how_cmd="")
            )
            for what in whats
        ]
        correlation = CorrelationDataset(datasets[0], datasets[-1])
        correlation.panel_label = "Correlation: " + " & ".join(
            [whats[0], whats[-1]]
        )
        return [correlation]

    @staticmethod
    def list_from_command(command):
        if command.what.is_combined:
            return DatasetFactory._list_from_combined(command)

        if command.when.is_interval:
            return DatasetFactory._list_from_interval(command)

        return [DatasetFactory.from_command(command)]
