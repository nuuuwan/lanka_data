"""Interactive console for querying lanka_data.

Usage:
    python workflows/console.py
"""

import json
import readline  # noqa: F401 — enables up-arrow history for input()

from rich.console import Console
from rich.prompt import Prompt

from lanka_data import db

console = Console()


def _query_and_print(path: str) -> None:
    try:
        result = db(path)
        console.print_json(json.dumps(result))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Error:[/bold red] {exc}")


def run() -> None:
    console.print(
        "[bold cyan]lanka_data console[/bold cyan]  "
        "(type [bold]exit[/bold] to quit)\n"
    )
    while True:
        path = Prompt.ask("[bold green]>[/bold green]").strip()
        if path.lower() in {"exit", "quit", "q"}:
            break
        if not path:
            continue
        _query_and_print(path)


if __name__ == "__main__":
    run()
