import os
import sys

from rich.console import Console

from lanka_data.console.ConsoleCommandLibrary import ConsoleCommandLibrary
from lanka_data.console.ConsoleCompleter import ConsoleCompleter
from lanka_data.console.ConsoleRenderer import ConsoleRenderer
from lanka_data.datasets.command.CommandRunner import CommandRunner


class ConsoleApp:
    def __init__(self, console=None, runner=None, library=None):
        self.console = console or Console()
        self.runner = runner or CommandRunner
        self.library = library or ConsoleCommandLibrary()
        self.renderer = ConsoleRenderer(self.console)
        self.completer = ConsoleCompleter(self.library.suggestions())

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
        self.completer.attach()
        self.renderer.show_banner()
        while True:
            try:
                line = input("lanka-data> ")
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break
            if not self.handle_line(line):
                break

    def handle_line(self, line):
        line = line.strip()
        if not line:
            return True
        if not self.handle_local_line(line):
            return False
        if line.lower() not in self.local_commands():
            self.run_command(line)
        return True

    def handle_local_line(self, line):
        line_lower = line.lower()
        if line_lower in ["exit", "quit"]:
            return False
        if line_lower == "clear":
            self.console.clear()
        elif line_lower == "help":
            self.renderer.show_help()
        elif line_lower == "fields":
            self.renderer.show_fields(self.library.field_rows())
        elif line_lower == "examples":
            self.renderer.show_examples(self.library.example_commands())
        elif line_lower == "commands":
            self.renderer.show_commands(self.library.command_suggestions())
        return True

    @staticmethod
    def local_commands():
        return ["clear", "help", "fields", "examples", "commands"]

    def run_command(self, line):
        command = line[4:].strip() if line.lower().startswith("run ") else line
        try:
            output = self.runner.run(command)
        except Exception as error:
            self.renderer.show_error(error)
            return
        self.open_image(output)
        self.renderer.show_output(output)

    @staticmethod
    def open_image(output):
        result = output.get("result")
        if not result or "image_path" not in result:
            return
        image_path = result["image_path"]
        if sys.platform == "darwin":
            os.system(f"open {image_path}")
