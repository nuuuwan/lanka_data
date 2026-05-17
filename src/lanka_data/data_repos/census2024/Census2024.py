import pathlib

from ...core.Query import Query
from ...core.Where import Where
from .Census2024ColRenamesMixin import Census2024ColRenamesMixin
from .Census2024DatasetsMixin import Census2024DatasetsMixin
from .Census2024FetchMixin import Census2024FetchMixin


class Census2024(
    Census2024DatasetsMixin,
    Census2024ColRenamesMixin,
    Census2024FetchMixin,
):
    _YEAR = "2024"
    _CACHE_DIR = pathlib.Path("/tmp/lanka_data")
    _tsv_cache: dict[str, list] = {}

    @classmethod
    def _label_has_entity(cls, label: str, where: Where) -> bool:
        if where.region_code == "*":
            return True
        for row in cls._fetch_tsv(label):
            if where.matches(row.get("region_id", "")):
                return True
        return False

    @classmethod
    def _query_label(cls, label: str, where: Where) -> dict:
        renames = cls._COL_RENAMES.get(label, {})
        result = {}
        for row in cls._fetch_tsv(label):
            eid = row.get("region_id", "")
            if not where.matches(eid):
                continue
            raw = cls._row_data(row)
            data = {renames.get(k, k): v for k, v in raw.items()}
            result[eid] = (
                next(iter(data.values())) if len(data) == 1 else data
            )
        if where.level is not None:
            return result
        return next(iter(result.values()), result)

    @classmethod
    def _entities(cls, label: str) -> dict:
        rows = cls._fetch_tsv(label)
        return {
            "entities": sorted(
                r["region_id"] for r in rows if r.get("region_id")
            )
        }

    @classmethod
    def _catalog(cls, q: Query) -> dict:
        if q.year is not None and q.year != cls._YEAR:
            return {"measurements": []}
        where = Where(q.where_raw)
        labels = [
            lbl for lbl in cls._DATASETS if cls._label_has_entity(lbl, where)
        ]
        return {"measurements": sorted(labels)}

    @classmethod
    def _wildcard_when(cls, q: Query) -> dict:
        if q.is_wildcard_where:
            return {"years": [cls._YEAR]}
        if cls._label_has_entity(q.what_raw, Where(q.where_raw)):
            return {"years": [cls._YEAR]}
        return {}

    @classmethod
    def _query_for_where(cls, q: Query) -> dict:
        if q.is_wildcard_where:
            return cls._entities(q.what_raw)
        return cls._query_label(q.what_raw, Where(q.where_raw))

    @classmethod
    def _query_for_time(cls, q: Query) -> dict:
        if q.is_wildcard_when:
            return cls._wildcard_when(q)
        if q.year != cls._YEAR:
            return {}
        return cls._query_for_where(q)

    @classmethod
    def query(cls, q: Query) -> dict:
        if q.is_wildcard_what:
            return cls._catalog(q)
        label = cls._resolve_label(q.what_raw)
        if label is None:
            return {}
        q.what_raw = label
        return cls._query_for_time(q)

    @classmethod
    def handles(cls, q: Query) -> bool:
        return cls._resolve_label(q.what_raw) is not None
