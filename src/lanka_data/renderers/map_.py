"""Choropleth tile-grid map SVG renderer for Sri Lanka regions."""
from ._palette import (
    PALETTE,
    _BG,
    _TITLE_COLOR,
    _LABEL_COLOR,
    _MUTED_COLOR,
    _FOOTER_COLOR,
    _escape,
    _svg_meta,
    _title_from_path,
    _footer_from_meta,
)

# ---------------------------------------------------------------------------
# Tile-grid positions  (row, col) for each LK district / province code
# ---------------------------------------------------------------------------

_DISTRICT_GRID: dict[str, tuple[int, int]] = {
    "LK-81": (0, 2),   # Jaffna
    "LK-83": (1, 1),   # Mannar
    "LK-82": (1, 2),   # Kilinochchi
    "LK-85": (1, 3),   # Mullaitivu
    "LK-84": (2, 2),   # Vavuniya
    "LK-42": (3, 0),   # Puttalam
    "LK-41": (3, 1),   # Kurunegala
    "LK-51": (3, 2),   # Anuradhapura
    "LK-92": (3, 3),   # Trincomalee
    "LK-12": (4, 1),   # Gampaha
    "LK-52": (4, 2),   # Polonnaruwa
    "LK-93": (4, 3),   # Batticaloa
    "LK-11": (5, 1),   # Colombo
    "LK-21": (5, 2),   # Kandy
    "LK-13": (6, 1),   # Kalutara
    "LK-22": (6, 2),   # Matale
    "LK-91": (6, 3),   # Ampara
    "LK-72": (7, 1),   # Kegalle
    "LK-23": (7, 2),   # Nuwara Eliya
    "LK-61": (7, 3),   # Badulla
    "LK-71": (8, 2),   # Ratnapura
    "LK-62": (8, 3),   # Moneragala
    "LK-31": (9, 2),   # Galle
    "LK-32": (10, 2),  # Matara
    "LK-33": (11, 2),  # Hambantota
}

_PROVINCE_GRID: dict[str, tuple[int, int]] = {
    "LK-7": (0, 1),   # Northern
    "LK-4": (1, 0),   # North Western
    "LK-5": (1, 1),   # North Central
    "LK-8": (1, 2),   # Eastern
    "LK-1": (2, 0),   # Western
    "LK-2": (2, 1),   # Central
    "LK-9": (2, 2),   # Uva
    "LK-6": (3, 0),   # Sabaragamuwa
    "LK-3": (3, 1),   # Southern
}

_DISTRICT_NAMES: dict[str, str] = {
    "LK-11": "Colombo",   "LK-12": "Gampaha",    "LK-13": "Kalutara",
    "LK-21": "Kandy",     "LK-22": "Matale",     "LK-23": "Nuwara Eliya",
    "LK-31": "Galle",     "LK-32": "Matara",     "LK-33": "Hambantota",
    "LK-41": "Kurunegala","LK-42": "Puttalam",
    "LK-51": "Anuradhapura", "LK-52": "Polonnaruwa",
    "LK-61": "Badulla",   "LK-62": "Moneragala",
    "LK-71": "Ratnapura", "LK-72": "Kegalle",
    "LK-81": "Jaffna",   "LK-82": "Kilinochchi","LK-83": "Mannar",
    "LK-84": "Vavuniya",  "LK-85": "Mullaitivu",
    "LK-91": "Ampara",    "LK-92": "Trincomalee","LK-93": "Batticaloa",
}

_PROVINCE_NAMES: dict[str, str] = {
    "LK-1": "Western",    "LK-2": "Central",   "LK-3": "Southern",
    "LK-4": "N.Western",  "LK-5": "N.Central", "LK-6": "Sabaragamuwa",
    "LK-7": "Northern",   "LK-8": "Eastern",   "LK-9": "Uva",
}

# ---------------------------------------------------------------------------
# Cell geometry
# ---------------------------------------------------------------------------

_CELL_W = 74
_CELL_H = 44
_GAP = 4
_STEP_X = _CELL_W + _GAP
_STEP_Y = _CELL_H + _GAP


def _get_name(code: str) -> str:
    return _DISTRICT_NAMES.get(code) or _PROVINCE_NAMES.get(code) or code


def _pick_grid(codes: list[str]) -> dict[str, tuple[int, int]] | None:
    if all(c in _DISTRICT_GRID for c in codes):
        return _DISTRICT_GRID
    if all(c in _PROVINCE_GRID for c in codes):
        return _PROVINCE_GRID
    return None


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------


def render_map(path: str, result: dict, meta: dict) -> str:
    from ..console.ConsoleFormatMixin import _is_total

    if not isinstance(result, dict):
        raise ValueError("Map requires a sub-region query.")
    for k in ("years", "entities", "measurements"):
        if k in result:
            raise ValueError("Map requires a breakdown response.")
    first = next(iter(result.values()), None)
    if not isinstance(first, dict):
        raise ValueError(
            "Map requires a sub-region query (e.g. LK:Districts). "
            "For single-region breakdown use Bar or Pie instead."
        )

    # Strip total keys; get non-total numeric breakdown per region
    stripped: dict[str, dict] = {}
    for code, sub in result.items():
        stripped[code] = {
            k: v
            for k, v in sub.items()
            if not _is_total(k) and isinstance(v, (int, float))
        }

    # Dominant category per region
    dominant: dict[str, str] = {}
    for code, vals in stripped.items():
        if vals:
            dominant[code] = max(vals, key=vals.__getitem__)

    # Stable color assignment sorted by category name
    categories = sorted(set(dominant.values()))
    cat_color = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(categories)}

    codes = list(result.keys())
    grid = _pick_grid(codes)

    if grid is not None:
        return _tile_svg(path, result, stripped, dominant, cat_color, categories, grid, meta)
    return _list_svg(path, result, stripped, dominant, cat_color, categories, meta)


# ---------------------------------------------------------------------------
# Tile-grid SVG
# ---------------------------------------------------------------------------


def _cell_svg(x: int, y: int, code: str, name: str, dom: str, color: str) -> str:
    short_dom = dom[:11] + ("…" if len(dom) > 11 else "")
    return (
        f'  <rect x="{x}" y="{y}" width="{_CELL_W}" height="{_CELL_H}" '
        f'fill="{color}" rx="4" opacity="0.88"/>\n'
        f'  <text x="{x + _CELL_W // 2}" y="{y + 12}" text-anchor="middle" '
        f'font-size="8" fill="#ffffffcc">{_escape(code)}</text>\n'
        f'  <text x="{x + _CELL_W // 2}" y="{y + 26}" text-anchor="middle" '
        f'font-size="11" font-weight="bold" fill="#ffffff">{_escape(name)}</text>\n'
        f'  <text x="{x + _CELL_W // 2}" y="{y + 39}" text-anchor="middle" '
        f'font-size="8" fill="#ffffffcc">{_escape(short_dom)}</text>'
    )


def _tile_svg(
    path, result, stripped, dominant, cat_color, categories, grid, meta
) -> str:
    codes = list(result.keys())
    max_row = max(grid[c][0] for c in codes if c in grid)
    max_col = max(grid[c][1] for c in codes if c in grid)

    grid_w = (max_col + 1) * _STEP_X - _GAP
    grid_h = (max_row + 1) * _STEP_Y - _GAP

    LEG_W = 220
    PAD_L = 40
    PAD_R = 20
    PAD_GAP = 30
    PAD_TOP = 62
    PAD_BOT = 44

    svg_w = PAD_L + grid_w + PAD_GAP + LEG_W + PAD_R
    svg_h = PAD_TOP + grid_h + PAD_BOT

    cells = []
    for code in codes:
        pos = grid.get(code)
        if pos is None:
            continue
        row, col = pos
        x = PAD_L + col * _STEP_X
        y = PAD_TOP + row * _STEP_Y
        dom = dominant.get(code, "")
        color = cat_color.get(dom, "#94a3b8")
        name = _get_name(code)
        cells.append(_cell_svg(x, y, code, name, dom, color))

    leg_x = PAD_L + grid_w + PAD_GAP
    leg_y0 = PAD_TOP
    legend = []
    for i, cat in enumerate(categories):
        y = leg_y0 + i * 28
        legend.append(
            f'  <rect x="{leg_x}" y="{y}" width="14" height="14" '
            f'fill="{cat_color[cat]}" rx="2"/>\n'
            f'  <text x="{leg_x + 20}" y="{y + 12}" font-size="12" '
            f'fill="{_LABEL_COLOR}">{_escape(cat)}</text>'
        )

    title = _escape(_title_from_path(path))
    footer = _escape(_footer_from_meta(meta))
    meta_block = _svg_meta(path, meta)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'font-family="system-ui,sans-serif">\n'
        f"{meta_block}\n"
        f'  <rect width="{svg_w}" height="{svg_h}" fill="{_BG}"/>\n'
        f'  <text x="{svg_w // 2}" y="42" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="{_TITLE_COLOR}">{title}</text>\n'
        f'{"".join(f"{c}" for c in cells)}\n'
        f'{"".join(legend)}\n'
        f'  <text x="{svg_w // 2}" y="{svg_h - 16}" text-anchor="middle" '
        f'font-size="11" fill="{_FOOTER_COLOR}">{footer}</text>\n'
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# Fallback: sorted coloured list (for unknown region codes such as EDs/GNDs)
# ---------------------------------------------------------------------------


def _list_svg(
    path, result, stripped, dominant, cat_color, categories, meta
) -> str:
    items = sorted(result.keys(), key=lambda c: dominant.get(c, ""))

    N_COLS = min(4, max(1, len(items) // 8 + 1))
    COL_W = 160
    CELL_H = 30
    n_rows = -(-len(items) // N_COLS)  # ceiling division

    PAD_TOP = 62
    PAD_BOT = 44 + 30 + ((-(-len(categories) // 5)) * 28)
    svg_w = max(700, N_COLS * COL_W + 80)
    svg_h = PAD_TOP + n_rows * CELL_H + PAD_BOT

    cells = []
    for i, code in enumerate(items):
        col = i // n_rows
        row = i % n_rows
        x = 40 + col * COL_W
        y = PAD_TOP + row * CELL_H
        dom = dominant.get(code, "")
        color = cat_color.get(dom, "#94a3b8")
        cells.append(
            f'  <rect x="{x}" y="{y}" width="{COL_W - 6}" height="24" '
            f'fill="{color}" rx="3" opacity="0.85"/>\n'
            f'  <text x="{x + 6}" y="{y + 16}" font-size="11" '
            f'fill="#ffffff" font-weight="bold">{_escape(code)}</text>'
        )

    leg_y0 = PAD_TOP + n_rows * CELL_H + 20
    legend = []
    for i, cat in enumerate(categories):
        col = i % 5
        row = i // 5
        x = 40 + col * 130
        y = leg_y0 + row * 28
        legend.append(
            f'  <rect x="{x}" y="{y}" width="12" height="12" '
            f'fill="{cat_color[cat]}" rx="2"/>\n'
            f'  <text x="{x + 16}" y="{y + 10}" font-size="11" '
            f'fill="{_LABEL_COLOR}">{_escape(cat[:14])}</text>'
        )

    title = _escape(_title_from_path(path))
    footer = _escape(_footer_from_meta(meta))
    meta_block = _svg_meta(path, meta)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'font-family="system-ui,sans-serif">\n'
        f"{meta_block}\n"
        f'  <rect width="{svg_w}" height="{svg_h}" fill="{_BG}"/>\n'
        f'  <text x="{svg_w // 2}" y="42" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="{_TITLE_COLOR}">{title}</text>\n'
        f'{"".join(cells)}\n'
        f'{"".join(legend)}\n'
        f'  <text x="{svg_w // 2}" y="{svg_h - 16}" text-anchor="middle" '
        f'font-size="11" fill="{_FOOTER_COLOR}">{footer}</text>\n'
        f"</svg>"
    )
