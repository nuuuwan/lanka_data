"""Bar chart SVG renderer."""
from ._palette import (
    PALETTE,
    _BG,
    _TITLE_COLOR,
    _LABEL_COLOR,
    _MUTED_COLOR,
    _FOOTER_COLOR,
    _escape,
    _svg_meta,
    _fmt_num,
    _title_from_path,
    _footer_from_meta,
)

_MAX_BARS = 20
_LABEL_W = 220
_BAR_MAX_W = 360
_NUM_W = 170
_WIDTH = _LABEL_W + _BAR_MAX_W + _NUM_W + 40  # 790
_ROW_H = 30
_PAD_TOP = 72
_PAD_BOT = 46


def render_bar(path: str, result: dict, meta: dict) -> str:
    from ..console.ConsoleFormatMixin import _is_total, _sort_by_val

    _validate_flat(result, "Bar")

    values = _sort_by_val(
        {
            k: v
            for k, v in result.items()
            if not _is_total(k) and isinstance(v, (int, float))
        }
    )
    if not values:
        raise ValueError("No numeric values to render as a bar chart.")

    total_raw = next(
        (v for k, v in result.items() if _is_total(k)), None
    )
    if not isinstance(total_raw, (int, float)):
        total_raw = sum(values.values()) or 1

    items = list(values.items())[:_MAX_BARS]
    max_val = max(v for _, v in items) or 1
    height = _PAD_TOP + len(items) * _ROW_H + _PAD_BOT

    rows = []
    for i, (label, val) in enumerate(items):
        y = _PAD_TOP + i * _ROW_H
        color = PALETTE[i % len(PALETTE)]
        bar_w = max(2, int(val / max_val * _BAR_MAX_W))
        pct = val / total_raw * 100 if total_raw else 0
        num_txt = _escape(f"{_fmt_num(val)} ({pct:.1f}%)")
        rows.append(
            f'  <text x="{_LABEL_W}" y="{y + 20}" text-anchor="end" '
            f'font-size="13" fill="{_LABEL_COLOR}">{_escape(label)}</text>\n'
            f'  <rect x="{_LABEL_W + 10}" y="{y + 6}" width="{bar_w}" height="18" '
            f'fill="{color}" rx="2"/>\n'
            f'  <text x="{_LABEL_W + 10 + bar_w + 6}" y="{y + 20}" '
            f'font-size="12" fill="{_MUTED_COLOR}">{num_txt}</text>'
        )

    title = _escape(_title_from_path(path))
    footer = _escape(_footer_from_meta(meta))
    meta_block = _svg_meta(path, meta)
    rows_svg = "\n".join(rows)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{height}" '
        f'font-family="system-ui,sans-serif">\n'
        f"{meta_block}\n"
        f'  <rect width="{_WIDTH}" height="{height}" fill="{_BG}"/>\n'
        f'  <text x="{_WIDTH // 2}" y="46" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="{_TITLE_COLOR}">{title}</text>\n'
        f"{rows_svg}\n"
        f'  <text x="{_WIDTH // 2}" y="{height - 16}" text-anchor="middle" '
        f'font-size="11" fill="{_FOOTER_COLOR}">{footer}</text>\n'
        f"</svg>"
    )


def _validate_flat(result: dict, renderer: str) -> None:
    if not isinstance(result, dict):
        raise ValueError(f"{renderer} requires a breakdown response.")
    for k in ("years", "entities", "measurements"):
        if k in result:
            raise ValueError(f"{renderer} requires a breakdown response.")
    first = next(iter(result.values()), None)
    if isinstance(first, dict):
        raise ValueError(
            f"{renderer} requires a single-region query. "
            "For sub-region breakdown use Map instead."
        )
