"""Color palette and shared SVG utilities."""
from datetime import datetime, timezone


class Palette:
    COLORS = [
        "#3b82f6",  # blue
        "#ef4444",  # red
        "#10b981",  # emerald
        "#f59e0b",  # amber
        "#8b5cf6",  # violet
        "#ec4899",  # pink
        "#06b6d4",  # cyan
        "#84cc16",  # lime
        "#f97316",  # orange
        "#6366f1",  # indigo
        "#14b8a6",  # teal
        "#a855f7",  # purple
    ]

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

    @staticmethod
    def title_from_path(path: str) -> str:
        parts = path.strip("/").split("/")
        return "  ·  ".join(parts[:3])

    @staticmethod
    def footer_from_meta(meta: dict) -> str:
        src = meta.get("source") or ""
        url = meta.get("source_url") or ""
        if src and url:
            return f"{src}  ·  {url}"
        return src or url
