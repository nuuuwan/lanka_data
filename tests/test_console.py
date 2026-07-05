from lanka_data.console.ConsoleApp import ConsoleApp
from lanka_data.console.ConsoleCommandLibrary import ConsoleCommandLibrary
from lanka_data.console.ConsoleCompleter import ConsoleCompleter
from lanka_data.console.ConsolePromptCompleter import ConsolePromptCompleter


class FakeRunner:
    command = None

    @classmethod
    def run(cls, command):
        cls.command = command
        return dict(command_str=command, result={}, sources=[])


class FakeDocument:
    def __init__(self, text):
        self.text_before_cursor = text


class TestConsole:
    def test_completer_matches_command_prefix(self):
        completer = ConsoleCompleter(["help", "Religion/2024/LK/JSON"])
        assert completer.find_matches("Rel") == ["Religion/2024/LK/JSON"]

    def test_completer_matches_later_terms(self):
        completer = ConsoleCompleter(
            ["Religion", "2024", "LK", "Religion/2024/LK/JSON"]
        )
        assert completer.find_matches("Religion/20") == ["Religion/2024"]
        assert completer.find_matches("Religion/2024/L") == [
            "Religion/2024/LK"
        ]

    def test_prompt_completer_yields_dropdown_completions(self):
        completer = ConsoleCompleter(["help", "Religion/2024/LK/JSON"])
        prompt_completer = ConsolePromptCompleter(completer)
        completions = list(
            prompt_completer.get_completions(FakeDocument("Rel"), None)
        )
        assert [c.text for c in completions] == ["Religion/2024/LK/JSON"]
        assert completions[0].start_position == -3

    def test_completer_uses_field_values_by_position(self):
        completer = ConsoleCompleter(
            ["Religion"],
            field_values=[
                ["Religion"],
                ["2024"],
                ["LK-1127025@20"],
                ["JSON", "Map:1st"],
            ],
        )
        matches = completer.find_matches("Religion/2024/LK-1127025@20/")
        assert matches == [
            "Religion/2024/LK-1127025@20/JSON",
            "Religion/2024/LK-1127025@20/Map:1st",
        ]
        assert "Religion/2024/LK-1127025@20/2024" not in matches

    def test_library_includes_meta_and_run_suggestions(self):
        library = ConsoleCommandLibrary(max_commands=1)
        suggestions = library.suggestions()
        assert "help" in suggestions
        assert any(value.startswith("run ") for value in suggestions)

    def test_app_runs_explicit_command(self):
        app = ConsoleApp(runner=FakeRunner)
        assert app.run_args(["run", "Help"]) == 0
        assert FakeRunner.command == "Help"

    def test_run_prefix_without_command_is_safe(self):
        app = ConsoleApp(runner=FakeRunner)
        assert app.normalize_command("run ") == ""
