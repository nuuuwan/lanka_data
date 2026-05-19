import urllib.request

from rich.console import Console as _RichConsole

_stderr = _RichConsole(stderr=True)

_META_COLS: frozenset[str] = frozenset(
    {
        "region_id",
        "region_name",
        "region_name_in_data",
        "region_ent_type",
    }
)


class Census2024FetchMixin:

    _BASE_DATA_URL = (
        "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data"
    )

    @staticmethod
    def _coerce(v: str):
        try:
            return int(v)
        except (ValueError, TypeError):
            pass
        try:
            return float(v)
        except (ValueError, TypeError):
            return v

    @classmethod
    def _load_tsv_text(cls, label: str) -> str:
        safe = label.replace(":", "_")
        cache_file = cls._CACHE_DIR / "census2024" / f"{safe}.tsv"
        if cache_file.exists():
            return cache_file.read_text()
        url = f"{cls._BASE_DATA_URL}/{cls._DATASETS[label]}"
        _stderr.print(
            f"[dim]  Downloading Census 2024 {label}...[/dim]", end=""
        )
        with urllib.request.urlopen(url) as r:
            text = r.read().decode()
        _stderr.print("[dim] done.[/dim]")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text)
        return text

    @staticmethod
    def _parse_tsv(text: str) -> list[dict]:
        lines = text.splitlines()
        headers = lines[0].split("\t")
        return [
            dict(zip(headers, line.split("\t"))) for line in lines[1:] if line
        ]

    @classmethod
    def _fetch_tsv(cls, label: str) -> list[dict]:
        if label not in cls._tsv_cache:
            cls._tsv_cache[label] = cls._parse_tsv(cls._load_tsv_text(label))
        return cls._tsv_cache[label]

    @classmethod
    def _row_data(cls, row: dict) -> dict:
        return {
            k: cls._coerce(v) for k, v in row.items() if k not in _META_COLS
        }
