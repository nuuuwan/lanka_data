"""Interactive console for querying lanka_data.

Usage:
    python workflows/console.py
"""

import json
import pathlib

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import completion_is_selected
from prompt_toolkit.history import FileHistory
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


def _meta_census2024(q) -> dict:
    file_path = Census2024._DATASETS.get(q.what_raw, "")
    repo_file = (
        f"{_CENSUS2024_RAW_BASE}/{file_path}" if file_path else None
    )
    return {
        "source": _SOURCE_CENSUS,
        "source_url": _SOURCE_CENSUS_URL,
        "repo_file": repo_file,
    }


def _meta_gig2(q) -> dict | None:
    norm_key, _ = q.gig2_key()
    entries = GIG2._build_index().get(norm_key or "", [])
    if not norm_key or not entries:
        return None
    is_election = norm_key.startswith("governmentelections")
    source = _SOURCE_ELECTION if is_election else _SOURCE_CENSUS
    source_url = (
        _SOURCE_ELECTION_URL if is_election else _SOURCE_CENSUS_URL
    )
    if q.is_wildcard_when:
        repo_file = "multiple"
    else:
        entry = next(
            (e for e in entries if e["year"] == q.year), entries[0]
        )
        repo_file = entry["url"]
    return {"source": source, "source_url": source_url, "repo_file": repo_file}


def _resolve_meta(q) -> dict:
    fallback = {
        "source": _SOURCE_CENSUS,
        "source_url": _SOURCE_CENSUS_URL,
        "repo_file": None,
    }
    if q.is_wildcard_what:
        return {
            "source": "multiple",
            "source_url": None,
            "repo_file": "multiple",
        }
    if Census2024.handles(q):
        return _meta_census2024(q)
    try:
        meta = _meta_gig2(q)
    except Exception:  # noqa: BLE001
        meta = None
    return meta or fallback


def _meta_for(path: str) -> dict:
    try:
        q = Query(path)
    except ValueError:
        return {"source": None, "source_url": None, "repo_file": None}
    return _resolve_meta(q)


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

    def _complete_what(self, raw: str, prefix: str):
        for token in _WHAT_COMPLETIONS:
            if token.lower().startswith(prefix.lower()):
                yield Completion(
                    "/" + token + "/",
                    start_position=-len(raw),
                    display="/" + token,
                )
        if "exit".startswith(prefix.lower()):
            yield Completion(
                "exit",
                start_position=-len(raw),
                display="/exit",
                display_meta="quit console",
            )

    def _complete_segment(self, parts: list, seg: int, prefix: str):
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

    def get_completions(self, document, complete_event):
        raw = document.text_before_cursor
        parts = raw.lstrip("/").split("/")
        seg = len(parts) - 1
        prefix = parts[-1]
        if seg == 0:
            yield from self._complete_what(raw, prefix)
        else:
            yield from self._complete_segment(parts, seg, prefix)


_kb = KeyBindings()


@_kb.add("enter", filter=completion_is_selected)
def _enter_accepts_completion(event):
    """Accept the highlighted completion without submitting the prompt."""
    buf = event.current_buffer
    buf.apply_completion(buf.complete_state.current_completion)


def _sort_by_val(d: dict) -> dict:
    """Sort a dict by numeric value descending."""
    return dict(
        sorted(
            d.items(),
            key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else -1,
            reverse=True,
        )
    )


def _pct_flat(d: dict) -> dict | None:
    """Percentages for a flat {field: numeric} dict, excluding 'Total'."""
    nums = {
        k: v
        for k, v in d.items()
        if k != "Total" and isinstance(v, (int, float))
    }
    raw_total = d.get("Total")
    total = (
        raw_total
        if isinstance(raw_total, (int, float))
        else sum(nums.values())
    )
    if not nums or not total:
        return None
    out = {}
    for k, v in nums.items():
        pct = v / total * 100
        if pct >= 0.005:
            out[k] = round(pct, 2)
    return _sort_by_val(out) or None


def _compute_p_values(result) -> dict | None:
    """Derive percentage breakdown from a query result."""
    special = ("years", "entities", "measurements")
    skip = not isinstance(result, dict) or any(k in result for k in special)
    if skip:
        return None
    first = next(iter(result.values()), None)
    if isinstance(first, dict):
        out = {k: _pct_flat(v) for k, v in result.items()}
        out = {k: v for k, v in out.items() if v is not None}
    else:
        out = _pct_flat(result)
    return out or None


def _entity_total(sub: dict):
    """Extract or compute the total for one entity sub-dict."""
    raw = sub.get("Total")
    if isinstance(raw, (int, float)):
        return raw
    nums = [
        v
        for k, v in sub.items()
        if k != "Total" and isinstance(v, (int, float))
    ]
    return sum(nums) if nums else None


def _total_and_strip(result) -> tuple:
    """Strip Total key; return (values, total_value, n_values)."""
    special = ("years", "entities", "measurements")
    if not isinstance(result, dict) or any(k in result for k in special):
        return result, None, None
    first = next(iter(result.values()), None)
    if isinstance(first, dict):
        total_value = {eid: _entity_total(sub) for eid, sub in result.items()}
        values = {
            eid: _sort_by_val({k: v for k, v in sub.items() if k != "Total"})
            for eid, sub in result.items()
        }
        return values, total_value, len(next(iter(values.values()), {}))
    raw = result.get("Total")
    nums = [
        v
        for k, v in result.items()
        if k != "Total" and isinstance(v, (int, float))
    ]
    if isinstance(raw, (int, float)):
        total_v = raw
    elif nums:
        total_v = sum(nums)
    else:
        total_v = None
    values = _sort_by_val({k: v for k, v in result.items() if k != "Total"})
    return values, total_v, len(values)


def _is_election(path: str) -> bool:
    """Return True if path resolves to an election dataset."""
    try:
        norm_key, _ = Query(path).gig2_key()
        return bool(norm_key and norm_key.startswith("governmentelections"))
    except Exception:  # noqa: BLE001
        return False


def _split_election(result: dict) -> tuple[dict, dict]:
    """Split an election result into (summary, party) dicts."""
    cols = GIG2._SUMMARY_COLS
    first = next(iter(result.values()), None)
    if isinstance(first, dict):
        summary = {
            eid: {k: v for k, v in sub.items() if k in cols}
            for eid, sub in result.items()
        }
        party = {
            eid: {k: v for k, v in sub.items() if k not in cols}
            for eid, sub in result.items()
        }
    else:
        summary = {k: v for k, v in result.items() if k in cols}
        party = {k: v for k, v in result.items() if k not in cols}
    return summary, party


def _query_and_print(path: str) -> None:
    try:
        result = db(path)
        meta = _meta_for(path)
        base = {
            "query": path,
            "source": meta["source"],
            "source_url": meta["source_url"],
            "repo_file": meta["repo_file"],
        }
        if _is_election(path):
            summary, party = _split_election(result)
            _, total_value, n_values = _total_and_strip(party)
            output = {
                **base,
                "summary": summary,
                "total_value": total_value,
                "n_values": n_values,
                "party": party,
                "p_party": _compute_p_values(party),
            }
        else:
            values, total_value, n_values = _total_and_strip(result)
            output = {
                **base,
                "total_value": total_value,
                "n_values": n_values,
                "values": values,
                "p_values": _compute_p_values(result),
            }
        console.print_json(json.dumps(output))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Error:[/bold red] {exc}")


def _handle_prompt_input(path: str) -> bool:
    """Process one prompt line; return False to exit the loop."""
    if path.lower() in {"exit", "/exit", "quit", "q"}:
        return False
    if path:
        _query_and_print(path)
    return True


def run() -> None:
    console.print(
        "[bold cyan]lanka_data console[/bold cyan]  "
        "(type [bold]exit[/bold] to quit, Tab to autocomplete)\n"
    )
    history_path = pathlib.Path.home() / ".lanka_data_history"
    session = PromptSession(
        completer=_PathCompleter(),
        complete_while_typing=True,
        key_bindings=_kb,
        history=FileHistory(str(history_path)),
    )
    while True:
        try:
            path = session.prompt("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not _handle_prompt_input(path):
            break


if __name__ == "__main__":
    run()
