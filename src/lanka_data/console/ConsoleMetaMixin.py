from lanka_data.core.Query import Query
from lanka_data.data_repos.census2024 import Census2024
from lanka_data.data_repos.gig2 import GIG2

_CENSUS2024_RAW_BASE = (
    "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data"
)

_SOURCE_CENSUS = "Department of Census and Statistics Sri Lanka"
_SOURCE_CENSUS_URL = "http://www.statistics.gov.lk/"
_SOURCE_ELECTION = "Election Commission of Sri Lanka"
_SOURCE_ELECTION_URL = "https://elections.gov.lk/"


class ConsoleMetaMixin:

    def _meta_census2024(self, q) -> dict:
        file_path = Census2024._DATASETS.get(q.what_raw, "")
        repo_file = (
            f"{_CENSUS2024_RAW_BASE}/{file_path}" if file_path else None
        )
        return {
            "source": _SOURCE_CENSUS,
            "source_url": _SOURCE_CENSUS_URL,
            "repo_file": repo_file,
        }

    def _meta_gig2(self, q) -> dict | None:
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
        return {
            "source": source,
            "source_url": source_url,
            "repo_file": repo_file,
        }

    def _resolve_meta(self, q) -> dict:
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
            return self._meta_census2024(q)
        try:
            meta = self._meta_gig2(q)
        except Exception:  # noqa: BLE001
            meta = None
        return meta or fallback

    def _meta_for(self, path: str) -> dict:
        try:
            q = Query(path)
        except ValueError:
            return {"source": None, "source_url": None, "repo_file": None}
        return self._resolve_meta(q)


def resolve_meta(path: str) -> dict:
    """Module-level helper: resolve source metadata for a query path."""
    return ConsoleMetaMixin()._meta_for(path)
