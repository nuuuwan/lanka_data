"""Horizontal bar chart SVG renderer."""

from .Palette import Palette

_MAX_BARS = 20
_LABEL_W = 220
_BAR_MAX_W = 360
_NUM_W = 170
_WIDTH = _LABEL_W + _BAR_MAX_W + _NUM_W + 40  # 790
_ROW_H = 30
_PAD_TOP = 72
_PAD_BOT = 46


class Bar:

    @staticmethod
    def validate_flat(result: dict, renderer: str) -> None:
        """Raise ValueError if result is not a flat numeric breakdown."""
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

    @staticmethod
    def render(path: str, result: dict, meta: dict) -> str:
        if not isinstance(result, dict):
            raise ValueError("Bar requires a breakdown response.")
        for k in ("years", "entities", "measurements"):
            if k in result:
                raise ValueError("Bar requires a breakdown response.")

        first = next(iter(result.values()), None)
        if isinstance(first, dict):
            return Bar._stacked_svg(path, result, meta)
        return Bar._flat_svg(path, result, meta)

    @staticmethod
    def _flat_svg(path: str, result: dict, meta: dict) -> str:
        from ..console.ConsoleFormatMixin import _is_total, _sort_by_val

        values = _sort_by_val(
            {
                k: v
                for k, v in result.items()
                if not _is_total(k) and isinstance(v, (int, float))
            }
        )
        if not values:
            raise ValueError("No numeric values to render as a bar chart.")

        total_raw = next((v for k, v in result.items() if _is_total(k)), None)
        if not isinstance(total_raw, (int, float)):
            total_raw = sum(values.values()) or 1

        items = list(values.items())[:_MAX_BARS]
        max_val = max(v for _, v in items) or 1
        height = _PAD_TOP + len(items) * _ROW_H + _PAD_BOT

        P = Palette
        rows = []
        for i, (label, val) in enumerate(items):
            y = _PAD_TOP + i * _ROW_H
            color = P.color_for(label, i)
            bar_w = max(2, int(val / max_val * _BAR_MAX_W))
            pct = val / total_raw * 100 if total_raw else 0
            num_txt = P.escape(f"{P.fmt_num(val)} ({pct:.1f}%)")
            rows.append(
                f'  <text x="{_LABEL_W}" y="{y + 20}" text-anchor="end" '
                f'font-size="13" fill="{
                    P.LABEL_COLOR}">{
                    P.escape(label)}</text>\n'
                f'  <rect x="{
                    _LABEL_W +
                    10}" y="{
                    y +
                    6}" width="{bar_w}" height="18" '
                f'fill="{color}" rx="2"/>\n'
                f'  <text x="{_LABEL_W + 10 + bar_w + 6}" y="{y + 20}" '
                f'font-size="12" fill="{P.MUTED_COLOR}">{num_txt}</text>'
            )

        title = P.escape(P.title_from_path(path))
        footer = P.escape(P.footer_from_meta(meta))
        meta_block = P.svg_meta(path, meta)
        rows_svg = "\n".join(rows)

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{height}" '
            f'font-family="system-ui,sans-serif">\n'
            f"{meta_block}\n"
            f'  <rect width="{_WIDTH}" height="{height}" fill="{P.BG}"/>\n'
            f'  <text x="{
                _WIDTH //
                2}" y="46" text-anchor="middle" font-size="16" '
            f'font-weight="bold" fill="{P.TITLE_COLOR}">{title}</text>\n'
            f"{rows_svg}\n"
            f'  <text x="{
                _WIDTH //
                2}" y="{
                height -
                16}" text-anchor="middle" '
            f'font-size="11" fill="{P.FOOTER_COLOR}">{footer}</text>\n'
            f"</svg>"
        )

    @staticmethod
    def _stacked_svg(path: str, result: dict, meta: dict) -> str:
        from ..console.ConsoleFormatMixin import _is_total

        P = Palette
        _SLABEL_W = 100
        _SBAR_W = 560
        _SROW_H = 26
        _SPAD_L = 16
        _SPAD_R = 16
        _SCOLS = 5

        # Strip totals per region
        regions_data = {
            region: {
                k: v
                for k, v in sub.items()
                if not _is_total(k) and isinstance(v, (int, float)) and v > 0
            }
            for region, sub in result.items()
            if isinstance(sub, dict)
        }
        regions_data = {r: v for r, v in regions_data.items() if v}
        if not regions_data:
            raise ValueError("No numeric values to render as a bar chart.")

        # Stable category → color assignment
        all_cats = sorted({c for vals in regions_data.values() for c in vals})
        cat_color = {
            cat: P.color_for(cat, i) for i, cat in enumerate(all_cats)
        }

        # Sort regions by total descending
        regions_sorted = sorted(
            regions_data.items(),
            key=lambda kv: sum(kv[1].values()),
            reverse=True,
        )

        leg_rows = -(-len(all_cats) // _SCOLS)
        svg_w = _SPAD_L + _SLABEL_W + _SBAR_W + _SPAD_R
        svg_h = (
            _PAD_TOP
            + len(regions_sorted) * _SROW_H
            + 30
            + leg_rows * 24
            + _PAD_BOT
        )

        rows = []
        for i, (region, vals) in enumerate(regions_sorted):
            y = _PAD_TOP + i * _SROW_H
            total = sum(vals.values()) or 1
            x = _SPAD_L + _SLABEL_W
            rows.append(
                f'  <text x="{
                    _SPAD_L +
                    _SLABEL_W -
                    6}" y="{
                    y +
                    _SROW_H //
                    2 +
                    4}" '
                f'text-anchor="end" font-size="11" fill="{P.LABEL_COLOR}">'
                f"{P.escape(region)}</text>"
            )
            for cat in all_cats:
                val = vals.get(cat, 0)
                if val <= 0:
                    continue
                seg_w = max(1, round(val / total * _SBAR_W))
                color = cat_color[cat]
                pct = val / total * 100
                rows.append(
                    f'  <rect x="{x}" y="{
                        y +
                        4}" width="{seg_w}" height="{
                        _SROW_H -
                        8}" '
                    f'fill="{color}" opacity="0.88">'
                    f"<title>{P.escape(region)} \u2014 {P.escape(cat)}: "
                    f"{P.fmt_num(val)} ({pct:.1f}%)</title>"
                    f"</rect>"
                )
                x += seg_w

        leg_y0 = _PAD_TOP + len(regions_sorted) * _SROW_H + 30
        col_w = (_SLABEL_W + _SBAR_W) // _SCOLS
        legend = []
        for i, cat in enumerate(all_cats):
            lx = _SPAD_L + (i % _SCOLS) * col_w
            ly = leg_y0 + (i // _SCOLS) * 24
            legend.append(
                f'  <rect x="{lx}" y="{ly}" width="12" height="12" '
                f'fill="{cat_color[cat]}" rx="2"/>\n'
                f'  <text x="{lx + 16}" y="{ly + 10}" font-size="11" '
                f'fill="{P.LABEL_COLOR}">{P.escape(cat[:20])}</text>'
            )

        title = P.escape(P.title_from_path(path))
        footer = P.escape(P.footer_from_meta(meta))
        meta_block = P.svg_meta(path, meta)

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
            f'font-family="system-ui,sans-serif">\n'
            f"{meta_block}\n"
            f'  <rect width="{svg_w}" height="{svg_h}" fill="{P.BG}"/>\n'
            f'  <text x="{
                svg_w // 2}" y="46" text-anchor="middle" font-size="16" '
            f'font-weight="bold" fill="{P.TITLE_COLOR}">{title}</text>\n'
            f'{"".join(rows)}\n'
            f'{"".join(legend)}\n'
            f'  <text x="{svg_w // 2}" y="{svg_h - 16}" text-anchor="middle" '
            f'font-size="11" fill="{P.FOOTER_COLOR}">{footer}</text>\n'
            f"</svg>"
        )
