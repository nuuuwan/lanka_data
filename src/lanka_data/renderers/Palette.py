"""Color palette and shared SVG utilities."""

from datetime import datetime, timezone


class Palette:
    COLORS = [
        # Sri Lankan flag colors
        "#8D153A",  # maroon (flag field)
        "#FF7000",  # saffron (left stripe)
        "#00534E",  # green (right stripe)
        "#FFD100",  # gold (border & lion)
        # Contrasting colors
        "#3b82f6",  # blue
        "#06b6d4",  # cyan
        "#8b5cf6",  # violet
        "#ec4899",  # pink
        "#10b981",  # emerald
        "#6366f1",  # indigo
        "#84cc16",  # lime
        "#14b8a6",  # teal
    ]

    # Semantic map checked by color_for() before falling back to COLORS.
    # Keys: lowercase, no spaces/hyphens/underscores.
    COLOR_MAP: dict[str, str] = {
        # ── Ethnicities (Sri Lankan flag symbolism) ──────────────────────
        "sinhalese": "#8D153A",  # maroon      – Sinhalese majority
        "srilankatamil": "#FF7000",  # saffron     – Sri Lankan Tamils
        "srilankatamils": "#FF7000",
        "indiantamil": "#C85000",  # deep saffron – Indian Tamils
        "indiantamils": "#C85000",
        "srilankanmoor": "#00534E",  # green       – Sri Lankan Moors
        "srilankanmoors": "#00534E",
        "veddha": "#8B5E3C",  # earthy brown – Veddha (indigenous)
        # ── Religions (Sri Lankan flag symbolism) ────────────────────────
        "buddhist": "#FFD100",  # gold        – Buddhism
        "buddhists": "#FFD100",
        "buddhism": "#FFD100",
        "hindu": "#FF7000",  # saffron     – Hinduism
        "hindus": "#FF7000",
        "hinduism": "#FF7000",
        "islam": "#00534E",  # green       – Islam
        "muslim": "#00534E",
        "muslims": "#00534E",
        "romancatholic": "#1A3A8F",  # deep blue   – Roman Catholic
        "christian": "#3b82f6",  # blue        – other Christian
        "christians": "#3b82f6",
        "christianity": "#3b82f6",
        # ── Political parties (official / widely-recognised colors) ──────
        "npp": "#CC0000",  # red        – National People's Power (JVP)
        "sjb": "#00843D",  # green      – Samagi Jana Balawegaya
        "unp": "#007A33",  # dark green – United National Party
        "slpp": "#003CA6",  # blue       – Sri Lanka Podujana Peramuna
        "slfp": "#003087",  # navy       – Sri Lanka Freedom Party
        "upfa": "#003087",  # navy       – United People's Freedom Alliance
        "itak": "#D62B2B",  # red        – Ilankai Tamil Arasu Kachchi
        "actc": "#FF6B00",  # orange-red – All Ceylon Tamil Congress
    }

    @classmethod
    def color_for(cls, label: str, index: int) -> str:
        """Return a semantic color for *label* if known, else COLORS[index]."""
        key = label.lower().replace(" ", "").replace("-", "").replace("_", "")
        return cls.COLOR_MAP.get(key, cls.COLORS[index % len(cls.COLORS)])

    BG = "#f9fafb"
    TITLE_COLOR = "#111827"
    LABEL_COLOR = "#374151"
    MUTED_COLOR = "#6b7280"
    FOOTER_COLOR = "#9ca3af"

    @staticmethod
    def escape(s) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def svg_meta(query: str, meta: dict) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        e = Palette.escape
        return (
            f"  <metadata>\n"
            f"    <query>{e(query)}</query>\n"
            f"    <source>{e(meta.get('source') or '')}</source>\n"
            f"    <source_url>{e(meta.get('source_url') or '')}</source_url>\n"
            f"    <repo_file>{e(meta.get('repo_file') or '')}</repo_file>\n"
            f"    <rendered>{ts}</rendered>\n"
            f"  </metadata>"
        )

    @staticmethod
    def fmt_num(v) -> str:
        if isinstance(v, float):
            return f"{v:,.2f}"
        return f"{v:,}"

    _LEVEL_SINGULAR: dict[str, str] = {
        "provinces": "Province",
        "districts": "District",
        "dsds": "DSD",
        "gnds": "GND",
        "electoraldistricts": "Electoral District",
        "eds": "Electoral District",
        "pollingdivisions": "Polling Division",
        "pds": "Polling Division",
    }

    @classmethod
    def _format_where(cls, where_raw: str) -> str:
        from ..data_repos.RegionNames import RegionNames

        if ":" in where_raw:
            code, level = where_raw.split(":", 1)
            region = (
                RegionNames.name_for(code) if code not in ("*", "") else "All"
            )
            level_label = cls._LEVEL_SINGULAR.get(level.lower(), level)
            return f"{region} by {level_label}"
        return RegionNames.name_for(where_raw)

    @classmethod
    def title_from_path(cls, path: str) -> str:
        parts = path.strip("/").split("/")
        what, when, where = parts[0], parts[1], parts[2]
        return "  ·  ".join([what, when, cls._format_where(where)])

    @staticmethod
    def footer_from_meta(meta: dict) -> str:
        src = meta.get("source") or ""
        url = meta.get("source_url") or ""
        if src and url:
            return f"{src}  ·  {url}"
        return src or url
