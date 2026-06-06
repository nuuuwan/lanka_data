import colorsys
import os
import random
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from lanka_data.how.GeoDataUtils import GeoDataUtils
from utils_future import Log

log = Log("MapUtils")

random.seed(0)


class COLOR:
    GOLD = "#FFBE29"
    DARK_ORANGE = "#EB7400"
    TEAL = "#00534E"
    BLUE = "#0000FF"
    MAROON = "#800080"
    GRAY = "#808080"
    DARK_RED = "#8D153A"
    GREEN = "#008000"
    RED = "#FF0000"
    ORANGE = "#FFA500"


class MapUtils:
    DELIM_TITLE = " · "
    MAX_REGIONS_TO_LABEL = 100
    MAX_LEGEND_ITEMS = 7
    COLOR_IDX = {
        # Religion
        "buddhist": COLOR.GOLD,
        "hindu": COLOR.DARK_ORANGE,
        "islam": COLOR.TEAL,
        "other_christian": COLOR.BLUE,
        "roman_catholic": COLOR.MAROON,
        "other": COLOR.GRAY,
        # Ethnicity
        "sinhalese": COLOR.DARK_RED,
        "sl_tamil": COLOR.DARK_ORANGE,
        "sri_lanka_tamil": COLOR.DARK_ORANGE,
        "ind_tamil": COLOR.BLUE,
        "indian_tamil_or_malaiyaga_thamilar": COLOR.BLUE,
        "sl_moor": COLOR.TEAL,
        "sri_lanka_moor_or_muslim": COLOR.TEAL,
        "malay": COLOR.GREEN,
        # Political Party
        "SLPP": COLOR.DARK_RED,
        "UPFA": COLOR.BLUE,
        "PA": COLOR.BLUE,
        "SLFP": COLOR.BLUE,
        "NPP": COLOR.RED,
        "SJB": COLOR.GREEN,
        "UNP": COLOR.GREEN,
        "NDF": COLOR.GREEN,
        "IND9": COLOR.ORANGE,
        "SLMP": COLOR.MAROON,
        "ACTC": COLOR.ORANGE,
        "ITAK": COLOR.ORANGE,
    }

    @staticmethod
    def get_random_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    @staticmethod
    def _color_with_opacity(hex_color, pct):
        """Return RGBA tuple for hex_color with alpha=pct (1.0=opaque, 0.0=transparent)."""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        MIN_ALPHA = 0.1
        alpha = MIN_ALPHA + pct * (1.0 - MIN_ALPHA)
        return (r, g, b, alpha)

    @staticmethod
    def get_colors_for_data_list_without_values(result_data):
        data_list = result_data["data_list"]
        n_regions = len(data_list)
        cmap = plt.cm.tab20  # pylint: disable=no-member.
        region_color_map = {
            data["region_id"]: cmap(i % 20)
            for i, data in enumerate(data_list[:n_regions])
        }
        value_to_color = None
        return region_color_map, value_to_color

    @staticmethod
    def get_func_key_getter(how, what):
        param = how.params
        if not param or param == "Top":
            idx = 0
        elif param == "2nd":
            idx = 1
        elif param == "3rd":
            idx = 2
        elif param == "Bottom":
            idx = -1
        else:
            return None

        def func_key_getter(data):
            return list(what.get_values(data).keys())[idx]

        return func_key_getter

    @staticmethod
    def get_colors_for_data_list_with_values_key(result_data, how, what):
        data_list = result_data["data_list"]
        value_to_color = {}
        region_color_map = {}
        pct_values = [data["pct_values"][how.params] for data in data_list]
        value_to_rank = {
            value: rank for rank, value in enumerate(sorted(set(pct_values)))
        }
        n = len(value_to_rank)
        for data in data_list:
            value = data["pct_values"][how.params]
            rank = value_to_rank[value]
            p = rank / (n - 1)
            hue = (1 - p) * 0.67
            sat = 1.0
            light = 0.5
            color = colorsys.hls_to_rgb(hue, light, sat)
            value_to_color[value] = color
            region_color_map[data["region_id"]] = color
        return region_color_map, value_to_color

    @staticmethod
    def get_colors_for_data_list_with_values_order(
        result_data, how, what, func_key_getter
    ):
        data_list = result_data["data_list"]
        key_to_base_hex = {}
        value_to_color = {}
        region_color_map = {}
        all_pcts = []
        raw_pcts = {}
        for data in data_list:
            key = func_key_getter(data) if func_key_getter else None
            if key not in key_to_base_hex:
                key_to_base_hex[key] = (
                    MapUtils.COLOR_IDX[key]
                    if key in MapUtils.COLOR_IDX
                    else MapUtils.get_random_color()
                )
                value_to_color[key] = MapUtils._color_with_opacity(
                    key_to_base_hex[key], 1.0
                )
            pct = (data.get("pct_values") or {}).get(key, 0.5)
            all_pcts.append(pct)
            raw_pcts[data["region_id"]] = (key, pct)

        pct_min = min(all_pcts)
        pct_max = max(all_pcts)
        pct_span = pct_max - pct_min if pct_max > pct_min else 1.0
        for region_id, (key, pct) in raw_pcts.items():
            normalised = (pct - pct_min) / pct_span
            region_color_map[region_id] = MapUtils._color_with_opacity(
                key_to_base_hex[key], normalised
            )
        value_to_color["__pct_range__"] = (pct_min, pct_max)
        return region_color_map, value_to_color

    @staticmethod
    def get_colors_for_data_list_with_values(result_data, how, what):
        func_key_getter = MapUtils.get_func_key_getter(how, what)
        if func_key_getter:
            return MapUtils.get_colors_for_data_list_with_values_order(
                result_data, how, what, func_key_getter
            )
        return MapUtils.get_colors_for_data_list_with_values_key(
            result_data,
            how,
            what,
        )

    @staticmethod
    def get_colors_for_data_list(result_data, how, what):
        data_list = result_data["data_list"]
        if what.get_values(data_list[0]) is None:
            return MapUtils.get_colors_for_data_list_without_values(
                result_data
            )
        return MapUtils.get_colors_for_data_list_with_values(
            result_data, how, what
        )  # region_id -> color

    @staticmethod
    def _is_light_color(color):
        if isinstance(color, str):
            color = color.lstrip("#")
            if len(color) == 6:
                r, g, b = (
                    int(color[0:2], 16) / 255,
                    int(color[2:4], 16) / 255,
                    int(color[4:6], 16) / 255,
                )
            else:
                return False
        else:
            # RGBA tuple: low alpha → near-transparent → treat as light
            if len(color) == 4 and color[3] < 0.4:
                return True
            r, g, b = color[0], color[1], color[2]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance > 0.5

    @staticmethod
    def _largest_polygon(geom):
        """Return the largest simple Polygon extracted from any geometry type."""
        from shapely.geometry import MultiPolygon, Polygon

        def _collect_polygons(g):
            if isinstance(g, Polygon):
                return [g]
            if hasattr(g, "geoms"):
                polys = []
                for sub in g.geoms:
                    polys.extend(_collect_polygons(sub))
                return polys
            return []

        polys = _collect_polygons(geom)
        if not polys:
            return geom
        return max(polys, key=lambda g: g.area)

    @staticmethod
    def _pole_of_inaccessibility(poly, n_cells=32):
        """Approximate the pole of inaccessibility (interior point farthest from
        the boundary) using a grid search.  This is a much better label anchor
        than the centroid for non-convex or L-shaped polygons like Ampara.
        """
        from shapely.geometry import Point

        minx, miny, maxx, maxy = poly.bounds
        boundary = poly.boundary
        best_dist = -1.0
        best_pt = poly.representative_point()
        xs = [
            minx + (maxx - minx) * (i + 0.5) / n_cells for i in range(n_cells)
        ]
        ys = [
            miny + (maxy - miny) * (j + 0.5) / n_cells for j in range(n_cells)
        ]
        for x in xs:
            for y in ys:
                pt = Point(x, y)
                if poly.contains(pt):
                    d = boundary.distance(pt)
                    if d > best_dist:
                        best_dist = d
                        best_pt = pt
        return best_pt

    @staticmethod
    def _best_label_fit(geom):
        """Search over rotation angles (0°–175° in 5° steps) to find the optimal
        rotated rectangle for label placement inside the largest polygon.

        The ray-casting origin is the pole of inaccessibility — the interior point
        farthest from the boundary — so non-convex regions like Ampara get a large
        rectangle rather than one squeezed against a nearby edge.

        For each candidate angle the polygon is rotated so that angle aligns with
        the x-axis and 4 axis-aligned rays are cast from that origin to measure
        the inscribed rectangle.  The rectangle is scored by the area of its
        intersection with the (rotated) polygon — penalising angles where the
        rectangle overflows non-convex voids.

        Returns (cx, cy, rect_w, rect_h, angle_deg) where angle_deg is the angle
        of the rectangle's long (x) axis in the original coordinate frame.
        """
        from shapely.affinity import rotate as shapely_rotate
        from shapely.geometry import LineString, Point, box

        poly = MapUtils._largest_polygon(geom)

        center_pt = MapUtils._pole_of_inaccessibility(poly)
        cx, cy = center_pt.x, center_pt.y

        b = poly.bounds
        span = max(b[2] - b[0], b[3] - b[1]) * 2

        best_score = -1.0
        best_result = (cx, cy, 0.0, 0.0, 0.0)

        N_ANGLES = 36  # 5° steps across 0°–175°

        for i in range(N_ANGLES):
            angle_deg = i * 180.0 / N_ANGLES

            # Rotate polygon so this angle aligns with the x-axis
            rpoly = shapely_rotate(poly, -angle_deg, origin=(cx, cy))
            rboundary = rpoly.boundary
            cp = Point(cx, cy)

            def ray(ddx, ddy, _rb=rboundary, _cp=cp):
                ln = LineString([(cx, cy), (cx + ddx * span, cy + ddy * span)])
                inter = _rb.intersection(ln)
                if inter.is_empty:
                    return span
                pts = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
                dists = [_cp.distance(p) for p in pts]
                return min(dists) if dists else span

            hw = min(ray(-1, 0), ray(1, 0))
            hh = min(ray(0, -1), ray(0, 1))
            rw, rh = 2 * hw, 2 * hh

            # Penalised score: area of the rectangle that lies inside the polygon
            rect_geom = box(cx - hw, cy - hh, cx + hw, cy + hh)
            score = rect_geom.intersection(rpoly).area

            if score > best_score:
                best_score = score
                best_result = (cx, cy, rw, rh, angle_deg)

        return best_result

    @staticmethod
    def _fit_fontsize(text, rect_w, rect_h, ax, fig):
        """Return font size (pts) so `text` fits inside a rect of rect_w × rect_h data units.
        rect_w is the along-text direction, rect_h is the across-text direction.
        """
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        frac_w = rect_w / (x_max - x_min)
        frac_h = rect_h / (y_max - y_min)

        avail_w_pts = frac_w * fig.get_figwidth() * 72
        avail_h_pts = frac_h * fig.get_figheight() * 72

        n_chars = max(len(text), 1)
        size_from_w = avail_w_pts / (n_chars * 0.6) * 0.3
        size_from_h = avail_h_pts / 1.2 * 0.3
        return max(3.0, min(size_from_w, size_from_h, 10.0))

    @staticmethod
    def _draw_labels(gdf_region, ax):
        fig = ax.get_figure()
        for _, row in gdf_region.iterrows():
            cx, cy, rect_w, rect_h, angle_deg = MapUtils._best_label_fit(
                row.geometry
            )
            bg_color = row.get("color", "black")
            text_color = (
                "black" if MapUtils._is_light_color(bg_color) else "white"
            )
            label = row.get("name", row["id"])

            # Orient text along the long axis of the rotated rectangle
            if rect_w >= rect_h:
                text_w, text_h = rect_w, rect_h
                text_angle = angle_deg
            else:
                text_w, text_h = rect_h, rect_w
                text_angle = angle_deg + 90.0

            # Normalise so text is never upside-down (keep in [-90°, 90°])
            while text_angle > 90.0:
                text_angle -= 180.0

            fontsize = MapUtils._fit_fontsize(label, text_w, text_h, ax, fig)
            ax.annotate(
                label,
                xy=(cx, cy),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
                rotation=text_angle,
            )

    @staticmethod
    def _format_legend_label(value):
        if isinstance(value, (int, float)):
            return f"{value:.1%}"
        return str(value)

    @staticmethod
    def _draw_legend_2d(value_to_color, legend_ax):

        MIN_ALPHA = 0.2
        pct_range = value_to_color.pop("__pct_range__", (0.0, 1.0))
        pct_min, pct_max = pct_range
        categories = sorted(value_to_color.keys(), key=str)
        n_rows = len(categories)
        pct_levels = [pct_min + i * (pct_max - pct_min) / 4 for i in range(5)]
        n_cols = len(pct_levels)

        # Build RGBA image: rows=categories, cols=pct levels
        img = np.zeros((n_rows, n_cols, 4))
        for row_i, cat in enumerate(reversed(categories)):
            r, g, b = value_to_color[cat][:3]
            for col_j, pct in enumerate(pct_levels):
                pct_span = pct_max - pct_min if pct_max > pct_min else 1.0
                normalised = (pct - pct_min) / pct_span
                img[row_i, col_j] = [
                    r,
                    g,
                    b,
                    MIN_ALPHA + normalised * (1.0 - MIN_ALPHA),
                ]

        # Constrain height so cells are square-ish; centre vertically
        cell_h = 0.12  # fraction of axes height per row
        grid_h = min(n_rows * cell_h, 0.9)
        y0 = (1.0 - grid_h) / 2
        inset = legend_ax.inset_axes([0.0, y0, 1.0, grid_h])
        # Hide the parent axes decorations but keep it alive for the inset
        legend_ax.set_axis_off()

        inset.imshow(img, aspect="auto", interpolation="nearest")

        # Column headers: percentage labels on top
        inset.set_xticks(range(n_cols))
        inset.set_xticklabels([f"{p:.0%}" for p in pct_levels], fontsize=5)
        inset.xaxis.set_ticks_position("top")
        inset.xaxis.set_label_position("top")
        inset.set_xlabel("% share", fontsize=5, labelpad=3)

        # Row labels: category names on left
        inset.set_yticks(range(n_rows))
        inset.set_yticklabels(
            [str(c) for c in reversed(categories)], fontsize=5
        )
        inset.tick_params(axis="both", length=0, pad=2)

        # White grid lines between cells
        for x in [i - 0.5 for i in range(n_cols + 1)]:
            inset.axvline(x, color="white", linewidth=0.5)
        for y in [i - 0.5 for i in range(n_rows + 1)]:
            inset.axhline(y, color="white", linewidth=0.5)

    @staticmethod
    def _draw_legend(value_to_color, ax, legend_ax):
        if value_to_color is None:
            legend_ax.set_visible(False)
            return

        colors = list(value_to_color.values())
        if colors and isinstance(colors[0], tuple) and len(colors[0]) == 4:
            MapUtils._draw_legend_2d(value_to_color, legend_ax)
            return

        legend_ax.set_visible(False)
        value_and_color = sorted(value_to_color.items(), reverse=True)
        if len(value_and_color) > MapUtils.MAX_LEGEND_ITEMS:
            n_actual = len(value_and_color)
            n_required = MapUtils.MAX_LEGEND_ITEMS - 1
            new_value_and_color = []
            for i in range(n_required):
                idx = int(i * n_actual / n_required)
                new_value_and_color.append(value_and_color[idx])
            new_value_and_color.append(value_and_color[-1])
            value_and_color = new_value_and_color

        for value, color in value_and_color:
            ax.scatter(
                [],
                [],
                color=color,
                label=MapUtils._format_legend_label(value),
            )
        ax.legend(fontsize=6)

    @staticmethod
    def draw_map(where, what, when, how):
        result_data = how.get_data(where, what, when)
        h = how.get_hash(where, what, when)

        data_list = result_data["data_list"]

        region_to_current_ids = {}
        for d in data_list:
            region_id = d["region_id"]
            current_ids = d.get("current_ids")
            if current_ids is None:
                current_ids = [region_id]
            region_to_current_ids[region_id] = current_ids

        region_ids = list(region_to_current_ids.keys())
        n_regions = len(region_ids)

        gdf_region = GeoDataUtils.get_geopandas_dataframe(
            region_to_current_ids
        )

        gdf_region = gdf_region.copy()
        region_color_map, value_to_color = MapUtils.get_colors_for_data_list(
            result_data, how, what
        )
        gdf_region["color"] = gdf_region["id"].map(region_color_map)

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        legend_ax = fig.add_subplot(gs[1])

        gdf_region.plot(
            ax=ax,
            categorical=True,
            color=gdf_region["color"],
            edgecolor="white",
            linewidth=0.2,
        )

        if n_regions <= MapUtils.MAX_REGIONS_TO_LABEL:
            MapUtils._draw_labels(gdf_region, ax)

        MapUtils._draw_legend(value_to_color, ax, legend_ax)
        title = MapUtils.DELIM_TITLE.join(
            how.get_title_items(where, what, when)
        )
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

        source = result_data.get("source", "")
        if source:
            fig.text(
                0.5,
                0.01,
                f"Source: {source}",
                ha="center",
                fontsize=7,
                color="gray",
            )

        image_dir = os.path.join(tempfile.gettempdir(), "lanka_data", "images")
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"{h}.png")
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        log.info(f"Wrote {image_path}")
        plt.close(fig)
        return {
            "image_path": image_path,
            "source": "Department of Census and Statistics, Sri Lanka",
            "source_url": "https://www.statistics.gov.lk/",
        }
