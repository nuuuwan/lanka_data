import importlib
import sys


class CompatibilityAliases:
    ALIASES = {
        "lanka_data.command": "lanka_data.datasets.command",
        "lanka_data.command.Command": "lanka_data.datasets.command.Command",
        "lanka_data.command.CommandBase": (
            "lanka_data.datasets.command.CommandBase"
        ),
        "lanka_data.command.CommandHelp": (
            "lanka_data.datasets.command.CommandHelp"
        ),
        "lanka_data.command.CommandIntrospectionMixin": (
            "lanka_data.datasets.command.CommandIntrospectionMixin"
        ),
        "lanka_data.command.CommandRunner": (
            "lanka_data.datasets.command.CommandRunner"
        ),
        "lanka_data.command.CommandCache": (
            "lanka_data.api.command.CommandCache"
        ),
        "lanka_data.command.CommandError": (
            "lanka_data.api.command.CommandError"
        ),
        "lanka_data.command.CommandLoaderMixin": (
            "lanka_data.api.command.CommandLoaderMixin"
        ),
        "lanka_data.command.InvalidCommandError": (
            "lanka_data.api.command.InvalidCommandError"
        ),
        "lanka_data.command.InvalidWhenError": (
            "lanka_data.api.command.InvalidWhenError"
        ),
        "lanka_data.command.InvalidWhereError": (
            "lanka_data.api.command.InvalidWhereError"
        ),
        "lanka_data.command.UnknownHowError": (
            "lanka_data.api.command.UnknownHowError"
        ),
        "lanka_data.command.UnknownWhatError": (
            "lanka_data.api.command.UnknownWhatError"
        ),
        "lanka_data.command.fields": "lanka_data.api.command.fields",
        "lanka_data.command.fields.How": (
            "lanka_data.api.command.fields.How"
        ),
        "lanka_data.command.fields.What": (
            "lanka_data.api.command.fields.What"
        ),
        "lanka_data.command.fields.When": (
            "lanka_data.api.command.fields.When"
        ),
        "lanka_data.command.fields.Where": (
            "lanka_data.api.command.fields.Where"
        ),
        "lanka_data.dataset": "lanka_data.datasets.dataset",
        "lanka_data.dataset.DatasetFactory": (
            "lanka_data.datasets.dataset.DatasetFactory"
        ),
        "lanka_data.dataset.EmptyDataset": (
            "lanka_data.datasets.dataset.EmptyDataset"
        ),
        "lanka_data.dataset.custom": "lanka_data.datasets.dataset.custom",
        "lanka_data.dataset.custom.Census2001Dataset": (
            "lanka_data.datasets.dataset.custom.Census2001Dataset"
        ),
        "lanka_data.dataset.custom.Census2012Dataset": (
            "lanka_data.datasets.dataset.custom.Census2012Dataset"
        ),
        "lanka_data.dataset.custom.Census2024Dataset": (
            "lanka_data.datasets.dataset.custom.Census2024Dataset"
        ),
        "lanka_data.dataset.custom.ElectionDataset": (
            "lanka_data.datasets.dataset.custom.ElectionDataset"
        ),
        "lanka_data.dataset.custom.ElectionSummaryDataset": (
            "lanka_data.datasets.dataset.custom.ElectionSummaryDataset"
        ),
        "lanka_data.dataset.custom.GIG2Dataset": (
            "lanka_data.datasets.dataset.custom.GIG2Dataset"
        ),
        "lanka_data.region": "lanka_data.datasets.region",
        "lanka_data.region.RegionParserMixin": (
            "lanka_data.datasets.region.RegionParserMixin"
        ),
        "lanka_data.region.RegionParserMixin.RegionParserMixin": (
            "lanka_data.datasets.region.RegionParserMixin.RegionParserMixin"
        ),
        "lanka_data.region.RegionParserMixin.RegionParserRadiusMixin": (
            "lanka_data.datasets.region.RegionParserMixin"
            ".RegionParserRadiusMixin"
        ),
    }

    @classmethod
    def register(cls):
        for alias, target in cls.ALIASES.items():
            cls._register(alias, target)

    @staticmethod
    def _register(alias, target):
        sys.modules.setdefault(alias, importlib.import_module(target))
