from lanka_data.console.ConsoleApp import ConsoleApp
from lanka_data.console.ConsoleCommandLibrary import ConsoleCommandLibrary
from lanka_data.console.ConsoleCompleter import ConsoleCompleter


class FakeRunner:
    command = None

    @classmethod
    def run(cls, command):
        cls.command = command
        return dict(command_str=command, result={}, sources=[])


class TestConsole:
    def test_completer_matches_command_prefix(self):
        completer = ConsoleCompleter(["help", "Religion/2024/LK/JSON"])
        assert completer.find_matches("Rel") == ["Religion/2024/LK/JSON"]

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
