"""Interactive console for querying lanka_data.

Usage:
    python workflows/console.py
"""

import json

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import completion_is_selected
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console

from lanka_data import db

console = Console()

_WHAT_COMPLETIONS = [
    # Census 2024
    "Housing",
    "AgeGroup",
    "Gender",
    "Households",
    "DrinkingWater",
    "CookingFuel",
    "Lighting",
    "Toilet",
    "Ethnicity",
    "Religion",
    # GIG2 domains and common sub-paths
    "Population",
    "Population:Ethnicity",
    "Population:AgeGroup",
    "Population:Gender",
    "Population:Religion",
    "Election",
    "Election:Presidential",
    "Election:General",
    "Economy",
    "Education",
    "Social",
    "*",
]

_WHERE_COMPLETIONS = [
    "LK",
    "LK:Provinces",
    "LK:Districts",
    "LK:DSDs",
    "LK:GNDs",
    "LK-1",
    "LK-2",
    "LK-3",
    "LK-4",
    "LK-5",
    "LK-6",
    "LK-7",
    "LK-8",
    "LK-9",
    "*",
]


class _PathCompleter(Completer):
    """Completes /<what>/<when>/<where> paths segment by segment."""

    _years_cache: dict[str, list[str]] = {}

    def _years_for(self, what: str) -> list[str]:
        if what not in self._years_cache:
            try:
                result = db(f"/{what}/*/LK")
                years = sorted(result.get("years", []))
            except Exception:  # noqa: BLE001
                years = []
            self._years_cache[what] = ["*"] + years
        return self._years_cache[what]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip("/")
        parts = text.split("/")
        seg = len(parts) - 1  # 0 = what, 1 = when, 2 = where
        prefix = parts[-1]

        if seg == 0:
            candidates = _WHAT_COMPLETIONS
        elif seg == 1:
            what = parts[0]
            candidates = self._years_for(what) if what and what != "*" else ["*"]
        elif seg == 2:
            candidates = _WHERE_COMPLETIONS
        else:
            return

        suffix = "/" if seg in (0, 1) else ""
        for token in candidates:
            if token.lower().startswith(prefix.lower()):
                yield Completion(
                    token + suffix,
                    start_position=-len(prefix),
                    display=token,
                )


_kb = KeyBindings()


@_kb.add("enter", filter=completion_is_selected)
def _enter_accepts_completion(event):
    """Accept the highlighted completion without submitting the prompt."""
    buf = event.current_buffer
    buf.apply_completion(buf.complete_state.current_completion)


def _query_and_print(path: str) -> None:
    try:
        result = db(path)
        console.print_json(json.dumps(result))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Error:[/bold red] {exc}")


def run() -> None:
    console.print(
        "[bold cyan]lanka_data console[/bold cyan]  "
        "(type [bold]exit[/bold] to quit, Tab to autocomplete)\n"
    )
    session = PromptSession(
        completer=_PathCompleter(),
        complete_while_typing=True,
        key_bindings=_kb,
    )
    while True:
        try:
            path = session.prompt("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if path.lower() in {"exit", "quit", "q"}:
            break
        if not path:
            continue
        _query_and_print(path)


if __name__ == "__main__":
    run()
