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


def _check_empty(result, q: Query, path: str) -> None:
    if result or q.is_wildcard_when or q.is_wildcard_where:
        return
    print(f"Warning: No data for {path!r}.", file=sys.stderr)


class Db:
    def __new__(cls, path: str = None):
        instance = object.__new__(cls)
        if path is not None:
            return instance(path)
        return instance

    def __call__(self, path: str) -> dict | str:
        q = Query(path)
        if q.is_wildcard_what:
            result = _merge_catalogs(GIG2.query(q), Census2024.query(q))
        elif Census2024.handles(q):
            gig2_r = GIG2.query(q)
            c24_r = Census2024.query(q)
            result = _merge_results(gig2_r, c24_r) if gig2_r else c24_r
        else:
            result = GIG2.query(q)
        _check_empty(result, q, path)

        if q.how == "JSON":
            return result

        # Visual renderers — imported lazily to avoid circular imports
        from ..console.ConsoleMetaMixin import resolve_meta
        from ..renderers import render_bar, render_map, render_pie

        meta = resolve_meta(path)
        if q.how == "Bar":
            return render_bar(path, result, meta)
        if q.how == "Pie":
            return render_pie(path, result, meta)
        return render_map(path, result, meta)
