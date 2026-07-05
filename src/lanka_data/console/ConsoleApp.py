import sys

from rich.console import Console

from lanka_data.console.ConsoleCommandLibrary import ConsoleCommandLibrary
from lanka_data.console.ConsoleCompleter import ConsoleCompleter
from lanka_data.console.ConsoleImageOpener import ConsoleImageOpener
from lanka_data.console.ConsoleLocalCommands import ConsoleLocalCommands
from lanka_data.console.ConsolePrompt import ConsolePrompt
from lanka_data.console.ConsoleRenderer import ConsoleRenderer
from lanka_data.datasets.command.CommandRunner import CommandRunner


class ConsoleApp:
    RUN_PREFIX = "run "

    def __init__(self, console=None, runner=None, library=None):
        self.console = console or Console()
        self.runner = runner or CommandRunner
        self.library = library or ConsoleCommandLibrary()
        self.renderer = ConsoleRenderer(self.console)
        self.completer = ConsoleCompleter(self.library.suggestions())
        self.prompt = ConsolePrompt(self.completer)
        self.local_commands = ConsoleLocalCommands(self)

    @classmethod
    def main(cls, args=None):
        return cls().run_args(sys.argv[1:] if args is None else args)

    def run_args(self, args):
        if args:
            line = " ".join(args)
            if line in ["-h", "--help"]:
                self.renderer.show_help()
                return 0
            self.handle_line(line)
            return 0
        self.run_loop()
        return 0

    def run_loop(self):
        self.renderer.show_banner()
        while True:
            if not self.read_next_line():
                break

    def read_next_line(self):
        line = self.safe_input()
        if line is None:
            return False
        return self.handle_line(line)

    def safe_input(self):
        try:
            return self.prompt.read()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return None

    def handle_line(self, line):
        line = line.strip()
        if not line:
            return True
        return self.dispatch_line(line)

    def dispatch_line(self, line):
        if not self.local_commands.handle(line):
            return False
        if line.lower() not in self.local_commands.names():
            self.run_command(line)
        return True

    def run_command(self, line):
        command = self.normalize_command(line)
        output = self.get_output(command)
        if output is None:
            return
        ConsoleImageOpener.open(output)
        self.renderer.show_output(output)

    def get_output(self, command):
        try:
            return self.runner.run(command)
        except Exception as error:
            self.renderer.show_error(error)
            return None

    def normalize_command(self, line):
        if line.lower().startswith(self.RUN_PREFIX):
            return self.normalize_run_command(line)
        return line

    @staticmethod
    def normalize_run_command(line):
        parts = line.split(maxsplit=1)
        if len(parts) == 1:
            return ""
        return parts[1].strip()
