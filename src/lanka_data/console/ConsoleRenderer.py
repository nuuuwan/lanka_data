import json

from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table


class ConsoleRenderer:
    BANNER_TEXT = (
        "Lanka Data Console\n"
        "Type help for commands. Suggestions appear as you type."
    )
    MAX_DISPLAYED_VALUES = 30

    def __init__(self, console):
        self.console = console

    def show_banner(self):
        self.console.print(Panel.fit(self.BANNER_TEXT, title="lanka_data"))

    def show_help(self):
        table = Table(title="Console commands")
        table.add_column("Command")
        table.add_column("Description")
        for command, description in self.help_rows():
            table.add_row(command, description)
        self.console.print(table)

    @staticmethod
    def help_rows():
        return [
            ("<cmd>", "Run an API command such as Religion/2024/LK/JSON"),
            ("run <cmd>", "Run an API command explicitly"),
            ("fields", "Show available what, when, where, and how values"),
            ("examples", "Show example commands"),
            ("commands", "Show generated valid commands"),
            ("clear", "Clear the terminal"),
            ("exit", "Exit the console"),
        ]

    def show_fields(self, rows):
        table = Table(title="API fields")
        table.add_column("Field")
        table.add_column("Values")
        for name, values in rows:
            table.add_row(
                name, ", ".join(values[: self.MAX_DISPLAYED_VALUES])
            )
        self.console.print(table)

    def show_examples(self, commands):
        self.show_command_table("Example commands", commands)

    def show_commands(self, commands):
        self.show_command_table("Valid commands", commands)

    def show_command_table(self, title, commands):
        table = Table(title=title)
        table.add_column("Command")
        self.add_command_rows(table, commands)
        self.console.print(table)

    @staticmethod
    def add_command_rows(table, commands):
        for command in commands:
            table.add_row(command)

    def show_output(self, output):
        text = json.dumps(output, indent=2)
        self.console.print(Syntax(text, "json", word_wrap=True))

    def show_error(self, error):
        self.console.print(f"[bold red]Error:[/bold red] {error}")
