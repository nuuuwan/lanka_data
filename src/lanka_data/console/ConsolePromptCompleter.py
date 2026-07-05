from prompt_toolkit.completion import Completer, Completion


class ConsolePromptCompleter(Completer):
    def __init__(self, completer):
        self.completer = completer

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        for match in self.completer.find_matches(text):
            yield Completion(match, start_position=-len(text))
