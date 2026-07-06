from lanka_data.api.command_errors.UnknownWhatError import UnknownWhatError
from lanka_data.api.dataset.CorrelationDataset import CorrelationDataset
from lanka_data.api.dataset.DiffDataset import DiffDataset
from lanka_data.api.dataset.SeriesDataset import SeriesDataset
from lanka_data.datasets.dataset.custom.Census2001Dataset import \
    Census2001Dataset
from lanka_data.datasets.dataset.custom.Census2012Dataset import \
    Census2012Dataset
from lanka_data.datasets.dataset.custom.Census2024Dataset import \
    Census2024Dataset
from lanka_data.datasets.dataset.custom.ElectionDataset import ElectionDataset
from lanka_data.datasets.dataset.custom.ElectionSummaryDataset import \
    ElectionSummaryDataset
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
        start = DatasetFactory.from_command(
            command.copy(when_cmd=years[0], how_cmd="")
        )
        end = DatasetFactory.from_command(
            command.copy(when_cmd=years[-1], how_cmd="")
        )
        return [DiffDataset(start, end)]

    @staticmethod
    def _list_from_series(command):
        years = command.when.years
        datasets = [
            DatasetFactory.from_command(
                command.copy(when_cmd=year, how_cmd="")
            )
            for year in years
        ]
        return [SeriesDataset(years, datasets)]

    @staticmethod
    def _list_from_combined(command):
        whats = command.what.whats
        first = DatasetFactory.from_command(
            command.copy(what_cmd=whats[0], how_cmd="")
        )
        last = DatasetFactory.from_command(
            command.copy(what_cmd=whats[-1], how_cmd="")
        )
        correlation = CorrelationDataset(first, last)
        correlation.panel_label = "Correlation: " + " & ".join(
            [whats[0], whats[-1]]
        )
        return [correlation]

    @staticmethod
    @staticmethod
    def _list_from_animation(command):
        datasets = []
        for year in command.when.years:
            dataset = DatasetFactory.from_command(
                command.copy(when_cmd=year, how_cmd="")
            )
            dataset.panel_label = year
            datasets.append(dataset)
        return datasets

    @staticmethod
    def _list_from_when(command):
        if command.how.needs_series:
            return DatasetFactory._list_from_series(command)
        return DatasetFactory._list_from_interval(command)

    @staticmethod
    def _list_default(command):
        return [DatasetFactory.from_command(command)]

    @staticmethod
    def _list_builder_rules():
        return [
            (
                lambda c: c.what.is_combined,
                DatasetFactory._list_from_combined,
            ),
            (
                lambda c: c.how.is_animation,
                DatasetFactory._list_from_animation,
            ),
            (lambda c: c.when.is_interval, DatasetFactory._list_from_when),
        ]

    @staticmethod
    def list_from_command(command):
        for matches, builder in DatasetFactory._list_builder_rules():
            if matches(command):
                return builder(command)
        return DatasetFactory._list_default(command)
