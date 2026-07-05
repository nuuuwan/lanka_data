class ConsoleCompleter:
    def __init__(self, suggestions):
        self.suggestions = suggestions
        self.matches = []

    def attach(self):
        try:
            import readline
        except ImportError:
            return False
        readline.set_completer(self.complete)
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("tab: complete")
        return True

    def complete(self, text, state):
        if state == 0:
            self.matches = self.find_matches(text)
        if state < len(self.matches):
            return self.matches[state]
        return None

    def find_matches(self, text):
        text_lower = text.lower()
        return [
            value
            for value in self.suggestions
            if value.lower().startswith(text_lower)
        ]
