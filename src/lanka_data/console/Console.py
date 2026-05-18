import json
import pathlib
import subprocess

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console as _RichConsole

from lanka_data.core import Db

from .ConsoleCacheMixin import ConsoleCacheMixin
from .ConsoleElectionMixin import ConsoleElectionMixin
from .ConsoleFormatMixin import ConsoleFormatMixin
from .ConsoleMetaMixin import ConsoleMetaMixin
from .PathCompleter import PathCompleter, _kb


class Console(
    ConsoleMetaMixin,
    ConsoleFormatMixin,
    ConsoleElectionMixin,
    ConsoleCacheMixin,
):
    def __init__(self) -> None:
        self.console = _RichConsole()

    @staticmethod
    def _svg_path(path: str) -> pathlib.Path:
        """Derive a filesystem path from a query path.

        '/Religion/2024/LK/Pie' → /tmp/lanka_data/images/Religion.2024.LK.Pie.svg
        """
        name = path.strip("/").replace("/", ".")
        out_dir = pathlib.Path("/tmp/lanka_data/images")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{name}.svg"

    def _query_and_print(self, path: str) -> None:
        try:
            result = Db(path)
            if isinstance(result, str):
                # SVG output — write to a named file and open with the default
                # app
                svg_path = self._svg_path(path)
                svg_path.write_text(result, encoding="utf-8")
                subprocess.run(["open", str(svg_path)], check=False)
                self.console.print(
                    f"[bold green]SVG saved:[/bold green] {svg_path}"
                )
                return
            meta = self._meta_for(path)
            base = {
                "query": path,
                "source": meta["source"],
                "source_url": meta["source_url"],
                "repo_file": meta["repo_file"],
            }
            if self._is_election(path):
                summary, party = self._split_election(result)
                _, total_value, n_values = self._total_and_strip(party)
                output = {
                    **base,
                    "summary": summary,
                    "total_value": total_value,
                    "n_values": n_values,
                    "party": party,
                    "p_party": self._compute_p_values(party),
                }
            else:
                values, total_value, n_values = self._total_and_strip(result)
                output = {
                    **base,
                    "total_value": total_value,
                    "n_values": n_values,
                    "values": values,
                    "p_values": self._compute_p_values(result),
                }
            self.console.print_json(json.dumps(output))
        except Exception as exc:  # noqa: BLE001
            self.console.print(f"[bold red]Error:[/bold red] {exc}")

    def _handle_prompt_input(self, path: str) -> bool:
        if path.lower() in {"exit", "/exit", "quit", "q"}:
            return False
        if path.lower() == "clear-cache":
            self._clear_cache()
        elif path:
            self._query_and_print(path)
        return True

    def run(self) -> None:
        self.console.print(
            "[bold cyan]lanka_data console[/bold cyan]  "
            "(type [bold]exit[/bold] to quit, Tab to autocomplete)\n"
        )
        history_path = pathlib.Path.home() / ".lanka_data_history"
        session = PromptSession(
            completer=PathCompleter(),
            complete_while_typing=True,
            key_bindings=_kb,
            history=FileHistory(str(history_path)),
        )
        while True:
            try:
                path = session.prompt("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not self._handle_prompt_input(path):
                break
