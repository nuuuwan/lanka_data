"""Pie chart SVG renderer."""
import math

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
from .bar import _validate_flat

_MAX_SLICES = 6
_CX, _CY, _R = 210, 215, 170
_WIDTH = 690
_HEIGHT = 460


def _arc_path(cx: float, cy: float, r: float, a1: float, a2: float) -> str:
    sweep = a2 - a1
    if sweep >= 2 * math.pi - 1e-4:
        # Full circle: draw two semicircles
        return (
            f"M {cx - r:.2f} {cy:.2f} "
            f"A {r} {r} 0 1 1 {cx + r:.2f} {cy:.2f} "
            f"A {r} {r} 0 1 1 {cx - r:.2f} {cy:.2f} Z"
        )
    x1 = cx + r * math.cos(a1)
    y1 = cy + r * math.sin(a1)
    x2 = cx + r * math.cos(a2)
    y2 = cy + r * math.sin(a2)
    large = 1 if sweep > math.pi else 0
    return (
        f"M {cx:.2f} {cy:.2f} "
        f"L {x1:.2f} {y1:.2f} "
        f"A {r} {r} 0 {large} 1 {x2:.2f} {y2:.2f} Z"
    )


def render_pie(path: str, result: dict, meta: dict) -> str:
    from ..console.ConsoleFormatMixin import _is_total, _sort_by_val

    _validate_flat(result, "Pie")

    values = _sort_by_val(
        {
            k: v
            for k, v in result.items()
            if not _is_total(k) and isinstance(v, (int, float))
        }
    )
    if not values:
        raise ValueError("No numeric values to render as a pie chart.")

    total_raw = next(
        (v for k, v in result.items() if _is_total(k)), None
    )
    if not isinstance(total_raw, (int, float)):
        total_raw = sum(values.values()) or 1

    items = list(values.items())
    if len(items) > _MAX_SLICES:
        top = items[:_MAX_SLICES]
        other_v = sum(v for _, v in items[_MAX_SLICES:])
        if other_v:
            top.append(("other", other_v))
        items = top

    total_items = sum(v for _, v in items) or 1

    slices = []
    angle = -math.pi / 2
    for i, (label, val) in enumerate(items):
        sweep = val / total_items * 2 * math.pi
        color = PALETTE[i % len(PALETTE)]
        slices.append((label, val, angle, angle + sweep, color))
        angle += sweep

    paths_svg = "\n".join(
        f'  <path d="{_arc_path(_CX, _CY, _R, a1, a2)}" fill="{color}" '
        f'stroke="white" stroke-width="1.5"/>'
        for _, _, a1, a2, color in slices
    )

    leg_x = _CX * 2 + 20
    leg_y0 = 70
    legend_rows = []
    for i, (label, val, _, _, color) in enumerate(slices):
        y = leg_y0 + i * 38
        pct = val / total_raw * 100
        legend_rows.append(
            f'  <rect x="{leg_x}" y="{y}" width="14" height="14" '
            f'fill="{color}" rx="2"/>\n'
            f'  <text x="{leg_x + 20}" y="{y + 12}" font-size="13" '
            f'fill="{_LABEL_COLOR}">{_escape(label)}</text>\n'
            f'  <text x="{leg_x + 20}" y="{y + 27}" font-size="11" '
            f'fill="{_MUTED_COLOR}">'
            f'{_escape(_fmt_num(val))} ({pct:.1f}%)</text>'
        )
    legend_svg = "\n".join(legend_rows)

    title = _escape(_title_from_path(path))
    footer = _escape(_footer_from_meta(meta))
    meta_block = _svg_meta(path, meta)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" '
        f'font-family="system-ui,sans-serif">\n'
        f"{meta_block}\n"
        f'  <rect width="{_WIDTH}" height="{_HEIGHT}" fill="{_BG}"/>\n'
        f'  <text x="{_WIDTH // 2}" y="44" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="{_TITLE_COLOR}">{title}</text>\n'
        f"{paths_svg}\n"
        f"{legend_svg}\n"
        f'  <text x="{_WIDTH // 2}" y="{_HEIGHT - 16}" text-anchor="middle" '
        f'font-size="11" fill="{_FOOTER_COLOR}">{footer}</text>\n'
        f"</svg>"
    )
