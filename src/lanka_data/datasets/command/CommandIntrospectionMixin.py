from lanka_data.datasets.command.fields import How, What, When, Where
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


class CommandIntrospectionMixin:
    @classmethod
    def field_classes(cls):
        return dict(what=What, when=When, where=Where, how=How)

    @classmethod
    def available_values(cls):
        return {
            name: field_cls.available_values()
            for name, field_cls in cls.field_classes().items()
        }

    @classmethod
    def describe_api(cls):
        return dict(
            format="<what>/<when>/<where>/<how>",
            fields={
                name: field_cls.describe()
                for name, field_cls in cls.field_classes().items()
            },
        )

    @classmethod
    def valid_commands(
        cls,
        where_values=None,
        when_values=None,
        how_values=None,
        max_commands=None,
    ):
        where_values = where_values or Where.available_values()
        when_values = when_values or When.available_values()
        how_values = how_values or How.available_values()
        commands = []
        for what, when in cls.valid_what_when_pairs(when_values):
            for where in where_values:
                for how in how_values:
                    cls.add_valid_command(
                        commands, what, when, where, how, max_commands
                    )
                    if max_commands and len(commands) >= max_commands:
                        return commands
        return commands

    @classmethod
    def valid_what_when_pairs(cls, when_values=None):
        pairs = cls.census_what_when_pairs()
        when_values = when_values or When.available_values()
        for label in ElectionDataset.get_labels():
            for when in when_values:
                pairs.append((label, when))
                pairs.append((label + "Summary", when))
        return sorted(set(pairs))

    @classmethod
    def census_what_when_pairs(cls):
        pairs = []
        for dataset_cls in cls.census_dataset_classes():
            for label in dataset_cls.get_labels():
                for when in dataset_cls.get_supported_whens():
                    pairs.append((label, when))
        return pairs

    @classmethod
    def census_dataset_classes(cls):
        return [
            Census2001Dataset,
            Census2012Dataset,
            Census2024Dataset,
        ]

    @classmethod
    def add_valid_command(
        cls, commands, what, when, where, how, max_commands
    ):
        if max_commands and len(commands) >= max_commands:
            return
        command_id = f"{what}/{when}/{where}/{how}"
        try:
            cls.from_str(command_id)
        except ValueError:
            return
        commands.append(command_id)
