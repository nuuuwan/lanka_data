"""Human-readable region name lookup from remote metadata files."""

import json
import pathlib
import re
import urllib.request

from rich.console import Console as _RichConsole

_stderr = _RichConsole(stderr=True)

_BASE_LK = (
    "https://raw.githubusercontent.com/"
    "nuuuwan/lk_admin_regions/main/data/ents"
)
_BASE_GIG = (
    "https://raw.githubusercontent.com/" "nuuuwan/gig-data/master/ents"
)

# Ordered from most-specific to least-specific pattern.
_URL_PATTERNS: list[tuple[str, str]] = [
    # Polling divisions:  EC-01A  (digits then a letter suffix)
    (r"^EC-\d+[A-Za-z]", f"{_BASE_GIG}/pd.tsv"),
    # Electoral districts: EC-01
    (r"^EC-", f"{_BASE_GIG}/ed.tsv"),
    # MOH areas
    (r"^MOH-", f"{_BASE_GIG}/moh.tsv"),
    # Local government
    (r"^LG-", f"{_BASE_GIG}/lg.tsv"),
    # LK province  – single digit:  LK-1 … LK-9
    (r"^LK-\d$", f"{_BASE_LK}/provinces.tsv"),
    # LK district  – two digits:    LK-11 … LK-92
    (r"^LK-\d{2}$", f"{_BASE_LK}/districts.tsv"),
    # LK DSD       – four digits:   LK-1101 …
    (r"^LK-\d{4}$", f"{_BASE_LK}/dsds.tsv"),
    # LK country
    (r"^LK$", f"{_BASE_LK}/countrys.tsv"),
]


class RegionNames:
    """Lazy, cached lookup of region codes → English names."""

    _CACHE_DIR = pathlib.Path("/tmp/lanka_data/ents")
    _cache: dict[str, str] = {}
    _loaded: set[str] = set()

    @classmethod
    def _fetch_and_cache(cls, url: str) -> None:
        if url in cls._loaded:
            return
        cls._loaded.add(url)  # mark now to avoid retrying on network error
        fname = url.rsplit("/", 1)[-1].replace(".tsv", ".json")
        cache_file = cls._CACHE_DIR / fname
        cls._CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if cache_file.exists():
            with cache_file.open() as f:
                cls._cache.update(json.load(f))
            return

        fname_short = url.rsplit("/", 1)[-1]
        _stderr.print(f"[dim]  Downloading {fname_short}...[/dim]", end="")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "lanka_data"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            _stderr.print("[dim] failed.[/dim]")
            return
        _stderr.print("[dim] done.[/dim]")

        lines = [ln for ln in content.splitlines() if ln.strip()]
        if not lines:
            return
        headers = lines[0].split("\t")
        try:
            id_i = headers.index("id")
            name_i = headers.index("name")
        except ValueError:
            return

        data: dict[str, str] = {}
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) > max(id_i, name_i):
                data[cols[id_i]] = cols[name_i]

        with cache_file.open("w") as f:
            json.dump(data, f)
        cls._cache.update(data)

    @classmethod
    def name_for(cls, code: str) -> str:
        """Return the English name for *code*, or *code* itself if unknown."""
        if code in cls._cache:
            return cls._cache[code]
        for pattern, url in _URL_PATTERNS:
            if re.match(pattern, code):
                cls._fetch_and_cache(url)
                return cls._cache.get(code, code)
        return code
