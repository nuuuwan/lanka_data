class ConsoleCompleter:
    def __init__(self, suggestions, field_values=None):
        self.suggestions = suggestions
        self.field_values = field_values or []
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
        separator = text.rfind("/")
        if separator == -1:
            return self._prefix_matches(self.suggestions, text)
        start = separator + 1
        prefix = text[:start]
        segment = text[start:]
        values = self.segment_values(prefix)
        matches = self._prefix_matches(values, segment)
        return [prefix + value for value in matches]

    def segment_values(self, prefix):
        index = prefix.count("/")
        if index < len(self.field_values):
            return self.field_values[index]
        return self.term_values()

    def term_values(self):
        return [value for value in self.suggestions if "/" not in value]

    @staticmethod
    def _prefix_matches(values, text):
        text_lower = text.lower()
        return [
            value for value in values if value.lower().startswith(text_lower)
        ]
