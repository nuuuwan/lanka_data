from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import completion_is_selected
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

from lanka_data.console.CompletionsData import CompletionsData
from lanka_data.core import Db


class PathCompleter(Completer):
    """Completes /<what>/<when>/<where> paths segment by segment."""

    _years_cache: dict[str, list[str]] = {}

    def _years_for(self, what: str) -> list[str]:
        if what not in self._years_cache:
            try:
                result = Db(f"/{what}/*/LK")
                years = sorted(result.get("years", []))
            except Exception:  # noqa: BLE001
                years = []
            self._years_cache[what] = ["*"] + years
        return self._years_cache[what]

    @staticmethod
    def _heading(label: str, noop_text: str, start: int) -> Completion:
        """Return a non-functional heading entry styled as a group title."""
        return Completion(
            noop_text,
            start_position=start,
            display=HTML(
                f"<b><ansibrightblack>── {label} ──</ansibrightblack></b>"
            ),
            display_meta="",
        )

    def _complete_what(self, raw: str, prefix: str):
        yield self._heading("‹what›", raw, -len(raw))
        for token in CompletionsData._WHAT_COMPLETIONS:
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
            )
        if "clear-cache".startswith(prefix.lower()):
            yield Completion(
                "clear-cache",
                start_position=-len(raw),
                display="clear-cache",
            )

    def _complete_segment(self, parts: list, seg: int, prefix: str):
        if seg == 1:
            what = parts[0]
            candidates = (
                self._years_for(what) if what and what != "*" else ["*"]
            )
            suffix = "/"
            meta = "<when>"
        elif seg == 2:
            candidates = CompletionsData._WHERE_COMPLETIONS
            suffix = "/"
            meta = "<where>"
        elif seg == 3:
            candidates = CompletionsData._HOW_COMPLETIONS
            suffix = ""
            meta = "<how>"
        else:
            return
        yield self._heading(f"‹{meta.strip('<>')}›", prefix, -len(prefix))
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
