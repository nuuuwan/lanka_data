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
from lanka_data.core.Query import Query
from lanka_data.data_repos.census2024 import Census2024
from lanka_data.data_repos.gig2 import GIG2

console = Console()

_CENSUS2024_RAW_BASE = (
    "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data"
)

_SOURCE_CENSUS = "Department of Census and Statistics Sri Lanka"
_SOURCE_CENSUS_URL = "http://www.statistics.gov.lk/"
_SOURCE_ELECTION = "Election Commission of Sri Lanka"
_SOURCE_ELECTION_URL = "https://elections.gov.lk/"


def _meta_for(path: str) -> dict:
    try:
        q = Query(path)
    except ValueError:
        return {"source": None, "source_url": None, "repo_file": None}

    if q.is_wildcard_what:
        return {
            "source": "multiple",
            "source_url": None,
            "repo_file": "multiple",
        }

    if Census2024.handles(q):
        file_path = Census2024._DATASETS.get(q.what_raw, "")
        repo_file = (
            f"{_CENSUS2024_RAW_BASE}/{file_path}" if file_path else None
        )
        return {
            "source": _SOURCE_CENSUS,
            "source_url": _SOURCE_CENSUS_URL,
            "repo_file": repo_file,
        }

    try:
        norm_key, _ = q.gig2_key()
        if norm_key:
            is_election = norm_key.startswith("governmentelections")
            source = _SOURCE_ELECTION if is_election else _SOURCE_CENSUS
            source_url = (
                _SOURCE_ELECTION_URL if is_election else _SOURCE_CENSUS_URL
            )
            index = GIG2._build_index()
            entries = index.get(norm_key, [])
            if entries:
                if q.is_wildcard_when:
                    return {
                        "source": source,
                        "source_url": source_url,
                        "repo_file": "multiple",
                    }
                entry = next(
                    (e for e in entries if e["year"] == q.year), entries[0]
                )
                return {
                    "source": source,
                    "source_url": source_url,
                    "repo_file": entry["url"],
                }
    except Exception:  # noqa: BLE001
        pass

    return {
        "source": _SOURCE_CENSUS,
        "source_url": _SOURCE_CENSUS_URL,
        "repo_file": None,
    }


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
        raw = document.text_before_cursor
        text = raw.lstrip("/")
        parts = text.split("/")
        seg = len(parts) - 1  # 0 = what, 1 = when, 2 = where
        prefix = parts[-1]

        if seg == 0:
            # For the first segment the user may have typed nothing, "/", or
            # "/El".  We replace the entire raw input so the inserted text
            # includes the leading slash.
            raw_prefix = raw
            candidates = _WHAT_COMPLETIONS
            for token in candidates:
                if token.lower().startswith(prefix.lower()):
                    yield Completion(
                        "/" + token + "/",
                        start_position=-len(raw_prefix),
                        display="/" + token,
                    )
            if "exit".startswith(prefix.lower()):
                yield Completion(
                    "exit",
                    start_position=-len(raw_prefix),
                    display="/exit",
                    display_meta="quit console",
                )
            return

        if seg == 1:
            what = parts[0]
            candidates = (
                self._years_for(what) if what and what != "*" else ["*"]
            )
            suffix = "/"
        elif seg == 2:
            candidates = _WHERE_COMPLETIONS
            suffix = ""
        else:
            return

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
        meta = _meta_for(path)
        output = {
            "query": path,
            "source": meta["source"],
            "source_url": meta["source_url"],
            "repo_file": meta["repo_file"],
            "results": result,
        }
        console.print_json(json.dumps(output))
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
        if path.lower() in {"exit", "/exit", "quit", "q"}:
            break
        if not path:
            continue
        _query_and_print(path)


if __name__ == "__main__":
    run()
