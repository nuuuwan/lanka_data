import sys

from ..data_repos.census2024 import Census2024
from ..data_repos.gig2 import GIG2
from .Query import Query


def _merge_catalogs(a: dict, b: dict) -> dict:
    m = set(a.get("measurements", []))
    n = set(b.get("measurements", []))
    return {"measurements": sorted(m | n)}


def _merge_results(a: dict, b: dict) -> dict:
    if "years" in a or "years" in b:
        years = sorted(set(a.get("years", [])) | set(b.get("years", [])))
        return {"years": years}
    if "entities" in a or "entities" in b:
        ents = sorted(set(a.get("entities", [])) | set(b.get("entities", [])))
        return {"entities": ents}
    return {**a, **b}


def _check_empty(result: dict, q: Query, path: str) -> None:
    if result or q.is_wildcard_when or q.is_wildcard_where:
        return
    print(f"Warning: No data for {path!r}.", file=sys.stderr)


class Db:
    def __call__(self, path: str) -> dict:
        q = Query(path)
        if q.is_wildcard_what:
            return _merge_catalogs(GIG2.query(q), Census2024.query(q))
        if Census2024.handles(q):
            result = _merge_results(GIG2.query(q), Census2024.query(q))
        else:
            result = GIG2.query(q)
        _check_empty(result, q, path)
        return result
