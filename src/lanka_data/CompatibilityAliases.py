import importlib
import sys


class CompatibilityAliases:
    COMMAND_DATASET_MODULES = [
        "CommandHelp",
        "CommandHelp.CommandHelp",
        "CommandRunner",
    ]
    COMMAND_API_MODULES = [
        "Command",
        "CommandBase",
        "CommandCache",
        "CommandLoaderMixin",
    ]
    COMMAND_ERROR_MODULES = [
        "CommandError",
        "InvalidCommandError",
        "InvalidWhenError",
        "InvalidWhereError",
        "UnknownHowError",
        "UnknownWhatError",
    ]
    FIELD_MODULES = ["How", "What", "When", "Where"]
    DATASET_MODULES = ["DatasetFactory", "EmptyDataset"]
    CUSTOM_MODULES = [
        "Census2001Dataset",
        "Census2012Dataset",
        "Census2024Dataset",
        "ElectionDataset",
        "ElectionSummaryDataset",
        "GIG2Dataset",
    ]
    REGION_MODULES = [
        "RegionParserMixin",
        "RegionParserMixin.RegionParserMixin",
        "RegionParserMixin.RegionParserRadiusMixin",
    ]

    @classmethod
    def register(cls):
        cls._register("lanka_data.command", "lanka_data.datasets.command")
        cls._register("lanka_data.dataset", "lanka_data.datasets.dataset")
        cls._register("lanka_data.region", "lanka_data.datasets.region")
        cls._register("lanka_data.command.fields", "lanka_data.api.fields")
        cls._register_command()
        cls._register_dataset()
        cls._register_many(
            cls.REGION_MODULES,
            "lanka_data.region",
            "lanka_data.datasets.region",
        )

    @classmethod
    def _register_command(cls):
        cls._register_many(
            cls.COMMAND_DATASET_MODULES,
            "lanka_data.command",
            "lanka_data.datasets.command",
        )
        cls._register_many(
            cls.COMMAND_API_MODULES,
            "lanka_data.command",
            "lanka_data.api.command",
        )
        cls._register_many(
            cls.COMMAND_ERROR_MODULES,
            "lanka_data.command",
            "lanka_data.api.command_errors",
        )
        cls._register_many(
            cls.FIELD_MODULES,
            "lanka_data.command.fields",
            "lanka_data.api.fields",
        )

    @classmethod
    def _register_dataset(cls):
        cls._register_many(
            cls.DATASET_MODULES,
            "lanka_data.dataset",
            "lanka_data.datasets.dataset",
        )
        cls._register(
            "lanka_data.dataset.custom", "lanka_data.datasets.dataset.custom"
        )
        cls._register_many(
            cls.CUSTOM_MODULES,
            "lanka_data.dataset.custom",
            "lanka_data.datasets.dataset.custom",
        )

    @classmethod
    def _register_many(cls, names, alias_prefix, target_prefix):
        for name in names:
            cls._register(f"{alias_prefix}.{name}", f"{target_prefix}.{name}")

    @staticmethod
    def _register(alias, target):
        sys.modules.setdefault(alias, importlib.import_module(target))


CompatibilityAliases.register()
