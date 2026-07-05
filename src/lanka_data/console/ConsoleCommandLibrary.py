from lanka_data.datasets.command.Command import Command
from lanka_data.api.command.fields import How, What, When, Where


class ConsoleCommandLibrary:
    DEFAULT_MAX_COMMANDS = 200
    META_COMMANDS = [
        "help",
        "fields",
        "examples",
        "commands",
        "clear",
        "exit",
        "quit",
    ]

    def __init__(self, max_commands=DEFAULT_MAX_COMMANDS):
        self.max_commands = max_commands

    def command_suggestions(self):
        commands = Command.valid_commands(max_commands=self.max_commands)
        return sorted(set(commands))

    def suggestions(self):
        values = self.META_COMMANDS + self.command_based_suggestions()
        values += self.field_based_suggestions()
        return sorted(set(values))

    def command_based_suggestions(self):
        commands = self.command_suggestions()
        return commands + [f"run {command}" for command in commands]

    @staticmethod
    def field_based_suggestions():
        return (
            What.available_values()
            + When.available_values()
            + When.available_intervals()
            + Where.available_values()
            + How.available_values()
        )

    def field_rows(self):
        return [
            ("what", What.available_values()),
            ("when", When.available_values() + When.available_intervals()),
            ("where", Where.available_values()),
            ("how", How.available_values()),
        ]

    def example_commands(self):
        commands = self.command_suggestions()
        return ["Help"] + commands[:10]
