from lanka_data.datasets.command.Command import Command
from lanka_data.datasets.command.fields import How, What, When, Where


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
        commands = self.command_suggestions()
        values = self.META_COMMANDS + commands
        values += [f"run {command}" for command in commands]
        values += What.available_values()
        values += When.available_values()
        values += When.available_intervals()
        values += Where.available_values()
        values += How.available_values()
        return sorted(set(values))

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
