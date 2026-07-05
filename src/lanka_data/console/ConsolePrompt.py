import sys

from prompt_toolkit import PromptSession

from lanka_data.console.ConsolePromptCompleter import ConsolePromptCompleter


class ConsolePrompt:
    MESSAGE = "lanka-data> "

    def __init__(self, completer):
        self.completer = completer
        self.session = self.build_session()

    def build_session(self):
        if not sys.stdin.isatty():
            self.completer.attach()
            return None
        try:
            return PromptSession(
                completer=ConsolePromptCompleter(self.completer),
                complete_while_typing=True,
            )
        except Exception:
            self.completer.attach()
            return None

    def read(self):
        if self.session is None:
            return input(self.MESSAGE)
        return self.session.prompt(self.MESSAGE)
