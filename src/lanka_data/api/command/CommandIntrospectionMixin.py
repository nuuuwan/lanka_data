from lanka_data.api.fields.How import How
from lanka_data.api.fields.What import What
from lanka_data.api.fields.When import When
from lanka_data.api.fields.Where import Where
from lanka_data.api.fields.WhatWhenRegistry import WhatWhenRegistry


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
        when_values = when_values or When.available_values()
        return WhatWhenRegistry.pairs(when_values)

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
