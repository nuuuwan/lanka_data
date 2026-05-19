"""Abstract base class for all data repository implementations."""

import pathlib
from abc import ABC, abstractmethod

from ..core.Query import Query


class AbstractDataRepo(ABC):
    """Common base for Census2024, GIG2, and future data repositories.

    Subclasses must implement:
        _data_query(cls, q)  – return raw JSON dict for a Query object
        handles(cls, q)      – return True if this repo covers the query
        _meta(cls, q)        – return source metadata dict for rendering

    The public API provided here:
        data_query(where, what, when)        – returns JSON dict
        query(where, what, when, how="JSON") – returns JSON or rendered SVG
    """

    _CACHE_DIR: pathlib.Path = pathlib.Path("/tmp/lanka_data")

    # ------------------------------------------------------------------
    # Abstract interface – subclasses must implement
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def _data_query(cls, q: Query) -> dict:
        """Return the raw data dict for the given Query."""
        ...

    @classmethod
    @abstractmethod
    def handles(cls, q: Query) -> bool:
        """Return True if this repository has data for the given Query."""
        ...

    @classmethod
    @abstractmethod
    def _meta(cls, q: Query) -> dict:
        """Return source metadata used by visual renderers.

        Expected keys: ``source``, ``source_url``, ``repo_file``.
        """
        ...

    # ------------------------------------------------------------------
    # Optional hook – override in subclasses when needed
    # ------------------------------------------------------------------

    @classmethod
    def _prerender_data(cls, q: Query, data: dict) -> dict:
        """Transform data before passing it to a visual renderer.

        The default implementation returns *data* unchanged.
        Override this to filter or reshape results for specific renderers
        (e.g. stripping summary columns for election visualisations).
        """
        return data

    # ------------------------------------------------------------------
    # Public API – fully implemented here, shared by all subclasses
    # ------------------------------------------------------------------

    @classmethod
    def _render(cls, path: str, how: str, data: dict, meta: dict) -> str:
        """Dispatch to the appropriate visual renderer."""
        # Lazy imports to avoid circular dependencies
        from ..renderers import Bar, Map, Pie  # noqa: PLC0415

        if how == "Bar":
            return Bar.render(path, data, meta)
        if how == "Pie":
            return Pie.render(path, data, meta)
        return Map.render(path, data, meta)

    @classmethod
    def data_query(cls, where: str, what: str, when: str) -> dict:
        """Return raw JSON data for the given where/what/when triple."""
        q = Query(f"/{where}/{what}/{when}")
        return cls._data_query(q)

    @classmethod
    def query(cls, where: str, what: str, when: str, how: str = "JSON"):
        """Return data in the requested format.

        ``how`` must be one of: JSON, Bar, Pie, Map (case-insensitive).
        When ``how`` is JSON (the default) a plain dict is returned.
        For visual formats the result is a rendered SVG string.
        """
        q = Query(f"/{where}/{what}/{when}/{how}")
        data = cls._data_query(q)
        if q.how == "JSON":
            return data
        data = cls._prerender_data(q, data)
        return cls._render(
            f"/{where}/{what}/{when}/{how}", q.how, data, cls._meta(q)
        )
