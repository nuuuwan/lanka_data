"""Choropleth map SVG renderer using real geographic boundaries."""

import json
import math
import os

from .Palette import Palette

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class Map:

    @staticmethod
    def _load_topo(name: str) -> dict:
        with open(os.path.join(_DATA_DIR, name), encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalize_pcode(raw: str) -> str:
        """Convert 'LK11' → 'LK-11', 'LK2' → 'LK-2'."""
        if raw and raw.startswith("LK") and "-" not in raw:
            return "LK-" + raw[2:]
        return raw

    @staticmethod
    def _decode_arcs(topo: dict) -> list:
        sx, sy = topo["transform"]["scale"]
        tx, ty = topo["transform"]["translate"]
        result = []
        for arc in topo["arcs"]:
            x = y = 0
            pts = []
            for point in arc:
                x += point[0]
                y += point[1]
                pts.append((x * sx + tx, y * sy + ty))
            result.append(pts)
        return result

    @staticmethod
    def _ring_coords(ring: list, decoded: list) -> list:
        coords = []
        for idx in ring:
            if idx >= 0:
                arc = decoded[idx]
                coords.extend(arc if not coords else arc[1:])
            else:
                arc = list(reversed(decoded[~idx]))
                coords.extend(arc if not coords else arc[1:])
        return coords

    @staticmethod
    def _feature_rings(geom: dict, decoded: list) -> list:
        if geom["type"] == "Polygon":
            return [Map._ring_coords(r, decoded) for r in geom["arcs"]]
        if geom["type"] == "MultiPolygon":
            rings = []
            for polygon in geom["arcs"]:
                rings.extend(Map._ring_coords(r, decoded) for r in polygon)
            return rings
        return []

    @staticmethod
    def _make_projector(bbox, mx, my, mw, mh):
        lon_min, lat_min, lon_max, lat_max = bbox
        cos_c = math.cos(math.radians((lat_min + lat_max) / 2))
        nat_w = (lon_max - lon_min) * cos_c
        nat_h = lat_max - lat_min
        scale = min(mw / nat_w, mh / nat_h)
        ox = mx + (mw - nat_w * scale) / 2
        oy = my + (mh - nat_h * scale) / 2

        def project(lon: float, lat: float):
            return (
                (lon - lon_min) * cos_c * scale + ox,
                (lat_max - lat) * scale + oy,
            )

        return project

    @staticmethod
    def _rings_to_d(rings, project) -> str:
        parts = []
        for ring in rings:
            if len(ring) < 2:
                continue
            px, py = project(*ring[0])
            pts = [f"M{px:.1f},{py:.1f}"]
            for lon, lat in ring[1:]:
                px, py = project(lon, lat)
                pts.append(f"L{px:.1f},{py:.1f}")
            pts.append("Z")
            parts.append("".join(pts))
        return " ".join(parts)

    @staticmethod
    def render(path: str, result: dict, meta: dict) -> str:
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

        stripped = {
            code: {
                k: v
                for k, v in sub.items()
                if not _is_total(k) and isinstance(v, (int, float))
            }
            for code, sub in result.items()
        }
        dominant = {
            code: max(vals, key=vals.__getitem__)
            for code, vals in stripped.items()
            if vals
        }
        categories = sorted(set(dominant.values()))
        cat_color = {
            cat: Palette.color_for(cat, i) for i, cat in enumerate(categories)
        }

        codes = list(result.keys())
        topo, pcode_field = Map._pick_topo(codes)
        if topo is not None:
            return Map._geo_svg(
                path, dominant, cat_color, categories, topo, pcode_field, meta
            )
        return Map._list_svg(
            path, result, dominant, cat_color, categories, meta
        )

    @staticmethod
    def _pick_topo(codes: list):
        for topo_name, pfield in [
            ("districts.topojson", "adm2_pcode"),
            ("provinces.topojson", "adm1_pcode"),
        ]:
            try:
                topo = Map._load_topo(topo_name)
            except FileNotFoundError:
                continue
            obj = next(iter(topo["objects"].values()))
            topo_codes = {
                Map._normalize_pcode(f["properties"].get(pfield, ""))
                for f in obj["geometries"]
            }
            if any(c in topo_codes for c in codes):
                return topo, pfield
        return None, None

    @staticmethod
    def _geo_svg(
        path: str,
        dominant: dict,
        cat_color: dict,
        categories: list,
        topo: dict,
        pcode_field: str,
        meta: dict,
    ) -> str:
        P = Palette
        PAD_L, PAD_R, PAD_TOP, PAD_BOT = 20, 20, 62, 44
        MAP_W, MAP_H = 360, 600
        LEG_W, LEG_GAP = 220, 24
        SVG_W = PAD_L + MAP_W + LEG_GAP + LEG_W + PAD_R
        SVG_H = PAD_TOP + MAP_H + PAD_BOT

        bbox = topo["bbox"]
        project = Map._make_projector(bbox, PAD_L, PAD_TOP, MAP_W, MAP_H)
        decoded = Map._decode_arcs(topo)
        obj = next(iter(topo["objects"].values()))
        name_field = (
            "adm2_name" if pcode_field == "adm2_pcode" else "adm1_name"
        )

        paths_svg = []
        for feat in obj["geometries"]:
            raw_code = feat["properties"].get(pcode_field, "")
            code = Map._normalize_pcode(raw_code)
            color = cat_color.get(dominant.get(code, ""), "#e5e7eb")
            rings = Map._feature_rings(feat, decoded)
            d = Map._rings_to_d(rings, project)
            if d:
                name = feat["properties"].get(name_field, "")
                dom = dominant.get(code, "")
                paths_svg.append(
                    f'  <path d="{d}" fill="{color}" stroke="white" '
                    f'stroke-width="0.7" opacity="0.88">'
                    f"<title>{P.escape(name)}: {P.escape(dom)}</title>"
                    f"</path>"
                )

        leg_x = PAD_L + MAP_W + LEG_GAP
        leg_y = PAD_TOP + 10
        leg_items = []
        for cat in categories:
            color = cat_color[cat]
            label = P.escape(cat[:28] + ("\u2026" if len(cat) > 28 else ""))
            leg_items.append(
                f'  <rect x="{leg_x}" y="{leg_y}" width="14" height="14" '
                f'fill="{color}" rx="2"/>\n'
                f'  <text x="{leg_x + 20}" y="{leg_y + 11}" font-size="12" '
                f'fill="{P.LABEL_COLOR}">{label}</text>'
            )
            leg_y += 22

        title = P.escape(P.title_from_path(path))
        footer = P.escape(P.footer_from_meta(meta))

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_W}" height="{SVG_H}" '
            f'font-family="system-ui,sans-serif">',
            P.svg_meta(path, meta),
            f'  <rect width="{SVG_W}" height="{SVG_H}" fill="{P.BG}"/>',
            f'  <text x="{PAD_L}" y="34" font-size="18" font-weight="bold" '
            f'fill="{P.TITLE_COLOR}">{title}</text>',
            f'  <rect x="{PAD_L}" y="{PAD_TOP}" width="{MAP_W}" height="{MAP_H}" '
            f'fill="#dbeafe" rx="4"/>',
        ]
        lines.extend(paths_svg)
        lines.extend(leg_items)
        lines.append(
            f'  <text x="{SVG_W // 2}" y="{SVG_H - 12}" text-anchor="middle" '
            f'font-size="11" fill="{P.FOOTER_COLOR}">{footer}</text>'
        )
        lines.append("</svg>")
        return "\n".join(lines)

    @staticmethod
    def _list_svg(path, result, dominant, cat_color, categories, meta) -> str:
        P = Palette
        items = sorted(result.keys(), key=lambda c: dominant.get(c, ""))
        N_COLS = min(4, max(1, len(items) // 8 + 1))
        COL_W = 160
        CELL_H = 30
        n_rows = -(-len(items) // N_COLS)
        PAD_TOP = 62
        legend_h = (-(-len(categories) // 5)) * 28
        PAD_BOT = 44 + 30 + legend_h
        svg_w = max(700, N_COLS * COL_W + 80)
        svg_h = PAD_TOP + n_rows * CELL_H + PAD_BOT
        cells = [f'  <rect x="{40 +
                          (i //
                           n_rows) *
                          COL_W}" y="{PAD_TOP +
                                      (i %
                                       n_rows) *
                                      CELL_H}" ' f'width="{
                COL_W -
                6}" height="24" fill="{
                cat_color.get(
                    dominant.get(
                        code,
                        ""),
                    "#94a3b8")}" ' f'rx="3" opacity="0.85"/>\n' f'  <text x="{40 + (i // n_rows) * COL_W + 6}" ' f'y="{PAD_TOP + (i % n_rows) * CELL_H + 16}" ' f'font-size="11" fill="#ffffff" font-weight="bold">{
                P.escape(code)}</text>' for i, code in enumerate(items)]
        leg_y0 = PAD_TOP + n_rows * CELL_H + 20
        legend = [
            f'  <rect x="{40 + (i % 5) * 130}" y="{leg_y0 + (i // 5) * 28}" '
            f'width="12" height="12" fill="{cat_color[cat]}" rx="2"/>\n'
            f'  <text x="{40 + (i %
                                5) * 130 + 16}" y="{leg_y0 + (i // 5) * 28 + 10}" '
            f'font-size="11" fill="{P.LABEL_COLOR}">{P.escape(cat[:14])}</text>'
            for i, cat in enumerate(categories)
        ]
        title = P.escape(P.title_from_path(path))
        footer = P.escape(P.footer_from_meta(meta))
        meta_block = P.svg_meta(path, meta)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
            f'font-family="system-ui,sans-serif">\n'
            f"{meta_block}\n"
            f'  <rect width="{svg_w}" height="{svg_h}" fill="{P.BG}"/>\n'
            f'  <text x="{
                svg_w // 2}" y="42" text-anchor="middle" font-size="16" '
            f'font-weight="bold" fill="{P.TITLE_COLOR}">{title}</text>\n'
            f'{"".join(cells)}\n'
            f'{"".join(legend)}\n'
            f'  <text x="{svg_w // 2}" y="{svg_h - 16}" text-anchor="middle" '
            f'font-size="11" fill="{P.FOOTER_COLOR}">{footer}</text>\n'
            f"</svg>"
        )
