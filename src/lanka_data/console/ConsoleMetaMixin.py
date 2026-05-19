from lanka_data.core.Query import Query
from lanka_data.data_repos.census2024 import Census2024
from lanka_data.data_repos.gig2 import GIG2

_FALLBACK_META = {
    "source": "Department of Census and Statistics Sri Lanka",
    "source_url": "http://www.statistics.gov.lk/",
    "repo_file": None,
}


class ConsoleMetaMixin:

    def _resolve_meta(self, q: Query) -> dict:
        if q.is_wildcard_what:
            return {
                "source": "multiple",
                "source_url": None,
                "repo_file": "multiple",
            }
        if Census2024.handles(q):
            return Census2024._meta(q)
        try:
            meta = GIG2._meta(q)
        except Exception:  # noqa: BLE001
            meta = None
        return meta or _FALLBACK_META

    def _meta_for(self, path: str) -> dict:
        try:
            q = Query(path)
        except ValueError:
            return {"source": None, "source_url": None, "repo_file": None}
        return self._resolve_meta(q)


def resolve_meta(path: str) -> dict:
    """Module-level helper: resolve source metadata for a query path."""
    return ConsoleMetaMixin()._meta_for(path)
