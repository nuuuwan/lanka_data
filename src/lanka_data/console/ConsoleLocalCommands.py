class ConsoleLocalCommands:
    EXIT_NAMES = ["exit", "quit"]
    NAMES = ["clear", "help", "fields", "examples", "commands"]

    def __init__(self, app):
        self.app = app

    def handle(self, line):
        line_lower = line.lower()
        if line_lower in self.EXIT_NAMES:
            return False
        actions = self.actions()
        if line_lower in actions:
            actions[line_lower]()
        return True

    @classmethod
    def names(cls):
        return cls.NAMES

    def actions(self):
        return dict(
            clear=self.app.console.clear,
            help=self.app.renderer.show_help,
            fields=self.show_fields,
            examples=self.show_examples,
            commands=self.show_commands,
        )

    def show_fields(self):
        self.app.renderer.show_fields(self.app.library.field_rows())

    def show_examples(self):
        self.app.renderer.show_examples(self.app.library.example_commands())

    def show_commands(self):
        self.app.renderer.show_commands(
            self.app.library.command_suggestions()
        )
