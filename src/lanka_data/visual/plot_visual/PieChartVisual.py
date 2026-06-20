import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
from matplotlib.ticker import FuncFormatter

from lanka_data.visual.plot.color_spec import ColorSpec, ColorSpecFactory
from lanka_data.visual.plot.Legend import Legend
from lanka_data.visual.plot_visual.PlotVisual import PlotVisual


class PieChartVisual(PlotVisual):
    # ── bar-chart constants/helpers (used when chart has negative values) ──

    BOTTOM_PADDING = 0.08

    @staticmethod
    def _is_change_chart(subregions):
        return any(
            value < 0
            for subregion in subregions
            for value in subregion["values"].values()
        )

    @classmethod
    def _sort_subregions(cls, subregions):
        if cls._is_change_chart(subregions):
            return sorted(
                subregions,
                key=lambda subregion: sum(subregion["values"].values()),
                reverse=True,
            )
        return sorted(
            subregions,
            key=lambda subregion: subregion.get(
                "total_value", sum(subregion["values"].values())
            ),
            reverse=True,
        )

    @staticmethod
    def _format_millions(value, _):
        label = "0"
        if value != 0:
            if abs(value) >= 1_000_000:
                label = f"{value / 1_000_000:.1f}M"
            elif abs(value) >= 1_000:
                label = f"{value / 1_000:.0f}K"
            elif abs(value) >= 10:
                label = f"{value:.0f}"
            else:
                label = f"{value:.2f}"
        return label

    @staticmethod
    def _draw_stacked_bars(
        ax,
        subregions,
        x_values,
        category_labels,
        category_to_color,
    ):
        y_max = 0
        y_min = 0
        for i, subregion in enumerate(subregions):
            x_value = x_values[i]
            values = subregion["values"]
            pos_bottom = 0
            neg_bottom = 0

            positive_categories = sorted(
                [
                    category
                    for category in category_labels
                    if values.get(category, 0) >= 0
                ],
                key=lambda category, value_map=values: value_map.get(
                    category, 0
                ),
            )
            negative_categories = sorted(
                [
                    category
                    for category in category_labels
                    if values.get(category, 0) < 0
                ],
                key=lambda category, value_map=values: value_map.get(
                    category, 0
                ),
                reverse=True,
            )

            for category in positive_categories:
                value = values.get(category, 0)
                if value == 0:
                    continue
                ax.bar(
                    [x_value],
                    [value],
                    bottom=[pos_bottom],
                    color=category_to_color[category],
                    label=category,
                    width=0.85,
                )
                pos_bottom += value

            for category in negative_categories:
                value = values.get(category, 0)
                ax.bar(
                    [x_value],
                    [value],
                    bottom=[neg_bottom],
                    color=category_to_color[category],
                    label=category,
                    width=0.85,
                )
                neg_bottom += value

            y_max = max(y_max, pos_bottom)
            y_min = min(y_min, neg_bottom)
        return y_min, y_max

    def _style_axis(self, ax, subregions, y_min, y_max, y_label="Population"):
        x_values = list(range(len(subregions)))
        ax.set_xticks(x_values)
        x_labels = [subregion["region_name"] for subregion in subregions]
        rotation = 90 if len(x_labels) > 12 else 45
        ax.set_xticklabels(
            x_labels,
            rotation=rotation,
            ha="right",
            fontsize=8,
        )
        pos = ax.get_position()
        padded_height = max(
            pos.height - self.BOTTOM_PADDING, pos.height * 0.7
        )
        ax.set_position(
            [
                pos.x0,
                pos.y0 + self.BOTTOM_PADDING,
                pos.width,
                padded_height,
            ]
        )
        ax.grid(False)
        ax.margins(x=0.06, y=0.12)
        y_span = y_max - y_min
        if y_span <= 0:
            y_span = 1
        y_pad = y_span * 0.12
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.axhline(0, color="#666", linewidth=0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(self._format_millions))
        ax.set_ylabel(y_label)

    @staticmethod
    def _draw_category_legend(ax, category_labels, category_to_color):
        legend_labels = category_labels[: min(len(category_labels), 10)]
        if legend_labels:
            value_to_color = {
                label: category_to_color[label]
                for label in legend_labels
                if label in category_to_color
            }
            if value_to_color:
                Legend.draw(value_to_color, ax)

    def _add_bar_labels(self, ax):
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height == 0:
                    continue
                p0 = ax.transData.transform((bar.get_x(), bar.get_y()))
                p1 = ax.transData.transform(
                    (
                        bar.get_x() + bar.get_width(),
                        bar.get_y() + height,
                    )
                )
                bar_h_px = abs(p1[1] - p0[1])
                bar_w_px = abs(p1[0] - p0[0])
                text = self._format_millions(height, None)
                n_chars = max(len(text), 1)
                bar_min_px = min(bar_w_px, bar_h_px)
                fontsize = min(
                    9,
                    bar_min_px * 0.6,
                    bar_h_px / (n_chars * 1.2),
                )
                if fontsize < 3:
                    continue
                x_c = bar.get_x() + bar.get_width() / 2
                y_c = bar.get_y() + height / 2
                fc = bar.get_facecolor()
                r, g, b = fc[0], fc[1], fc[2]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "#333" if lum > 0.5 else "#eee"
                ax.text(
                    x_c,
                    y_c,
                    text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=text_color,
                    rotation=90,
                    clip_on=True,
                )

    # ── pie-chart helpers ──────────────────────────────────────────────────

    @staticmethod
    def _get_max_abs_category_change(subregions):
        max_abs_change = 0
        for subregion in subregions:
            for value in subregion["values"].values():
                max_abs_change = max(max_abs_change, abs(value))
        return max(max_abs_change, 1)

    @staticmethod
    def _distance_to_bounds(center, bounds):
        x_min, y_min, x_max, y_max = bounds
        x, y = center
        return min(x - x_min, x_max - x, y - y_min, y_max - y)

    @staticmethod
    def _pair_radius_cap(center, all_centers, fallback):
        min_dist = None
        for other_center in all_centers:
            if other_center == center:
                continue
            dx = center[0] - other_center[0]
            dy = center[1] - other_center[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if min_dist is None or dist < min_dist:
                min_dist = dist
        if min_dist is None:
            return fallback
        return min_dist / 2

    @classmethod
    def _get_change_bar_geometry(cls, center, all_centers, bounds, span_min):
        pair_cap = cls._pair_radius_cap(
            center,
            all_centers,
            span_min * 0.1,
        )
        bound_cap = cls._distance_to_bounds(center, bounds)
        bar_half_width = max(
            min(pair_cap * 0.25, span_min * 0.03),
            span_min * 0.006,
        )
        bar_max_height = max(
            min(pair_cap * 0.95, bound_cap * 0.9, span_min * 0.2),
            span_min * 0.01,
        )
        return bar_half_width, bar_max_height

    @classmethod
    def _draw_change_bar_for_subregion(cls, ax, subregion, center, draw_ctx):
        bar_half_width = draw_ctx["bar_half_width"]
        bar_max_height = draw_ctx["bar_max_height"]
        max_abs_net_change = draw_ctx["max_abs_net_change"]
        category_labels = draw_ctx["category_labels"]
        category_to_color = draw_ctx["category_to_color"]
        span_min = draw_ctx["span_min"]

        x_center, y_center = center
        n_categories = max(len(category_labels), 1)
        full_width = 2 * bar_half_width
        slot_width = full_width / n_categories
        bar_width = slot_width * 0.92
        min_label_y = y_center
        for category_index, category in enumerate(category_labels):
            value = subregion["values"].get(category, 0)
            height = abs(value) / max_abs_net_change * bar_max_height
            x0 = (
                x_center
                - bar_half_width
                + slot_width * category_index
                + (slot_width - bar_width) / 2
            )
            y0 = y_center if value >= 0 else y_center - height
            rect = Rectangle(
                (x0, y0),
                bar_width,
                height,
                facecolor=category_to_color.get(category, "#999"),
                edgecolor="#fff",
                linewidth=0.2,
            )
            ax.add_patch(rect)
            min_label_y = min(min_label_y, y0)

        ax.plot(
            [x_center - bar_half_width, x_center + bar_half_width],
            [y_center, y_center],
            color="#444",
            linewidth=0.7,
        )
        label_y = min_label_y - span_min * 0.008
        ax.text(
            x_center,
            label_y,
            f"{subregion['region_name']}",
            ha="center",
            va="top",
            fontsize=6,
            color="#111",
        )

    @staticmethod
    def _format_population(total_value):
        label = ""
        if total_value is not None:
            if total_value >= 1_000_000:
                label = f"{total_value / 1_000_000:.2f}M"
            elif total_value >= 1_000:
                label = f"{total_value / 1_000:.1f}K"
            else:
                label = f"{total_value:.0f}"
        return label

    @staticmethod
    def _order_positive_values_with_top_first(values_map):
        items = [
            (label, value) for label, value in values_map.items() if value > 0
        ]
        if not items:
            return []
        top_label, _ = max(items, key=lambda item: item[1])
        return [
            (label, value) for label, value in items if label == top_label
        ] + [(label, value) for label, value in items if label != top_label]

    @staticmethod
    def _get_startangle_with_top_half(top_value, total_value):
        if total_value <= 0:
            return 90
        top_angle = 360.0 * top_value / total_value
        return 90 + top_angle / 2

    @classmethod
    def _draw_pie_at(cls, ax, center, radius, values_map, color_map):
        ordered_items = cls._order_positive_values_with_top_first(values_map)
        total_value = sum(value for _, value in ordered_items)
        if total_value <= 0:
            return
        top_value = ordered_items[0][1]
        theta = cls._get_startangle_with_top_half(top_value, total_value)
        for label, value in ordered_items:
            angle = 360.0 * value / total_value
            next_theta = theta - angle
            wedge = Wedge(
                center,
                radius,
                next_theta,
                theta,
                facecolor=color_map.get(label, "#999"),
                edgecolor="#fff",
                linewidth=0.2,
            )
            ax.add_patch(wedge)
            theta = next_theta

    @classmethod
    def _draw_map_centered_pies(cls, ax, chart_data):
        subregions = chart_data["subregions"]
        centers = chart_data["centers"]
        bounds = chart_data["bounds"]
        category_to_color = chart_data["category_to_color"]
        gdf = chart_data["gdf"]

        if gdf is not None and not gdf.empty:
            gdf.boundary.plot(ax=ax, color="#999", linewidth=0.4, alpha=0.4)

        x_min, y_min, x_max, y_max = bounds
        span_x = max(x_max - x_min, 1e-9)
        span_y = max(y_max - y_min, 1e-9)
        span_min = min(span_x, span_y)

        max_total = max(
            [max(subregion["total_value"], 0) for subregion in subregions]
            + [1]
        )
        min_radius = span_min * 0.015
        max_radius = span_min * 0.08
        radius_gap = span_min * 0.004
        all_centers = list(centers.values())

        for subregion in subregions:
            center = centers.get(subregion["region_id"])
            if center is None:
                continue
            total_value = max(subregion["total_value"], 0)
            scale = math.sqrt(total_value / max_total) if max_total > 0 else 0
            base_radius = min_radius + (max_radius - min_radius) * scale
            pair_cap = cls._pair_radius_cap(center, all_centers, max_radius)
            bound_cap = cls._distance_to_bounds(center, bounds)
            radius = min(
                base_radius,
                max(pair_cap - radius_gap, 0),
                max(bound_cap - radius_gap, 0),
            )
            if radius <= 0:
                continue
            cls._draw_pie_at(
                ax, center, radius, subregion["values"], category_to_color
            )
            ax.text(
                center[0],
                center[1] - radius - span_min * 0.01,
                (
                    f"{subregion['region_name']}\n"
                    f"{cls._format_population(subregion['total_value'])}"
                ),
                ha="center",
                va="top",
                fontsize=6,
                color="#111",
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

        category_labels = chart_data["category_labels"]
        legend_items = category_labels[: min(len(category_labels), 10)]
        if legend_items:
            value_to_color = {
                label: category_to_color[label]
                for label in legend_items
                if label in category_to_color
            }
            Legend.draw(value_to_color, ax)

    @classmethod
    def _draw_map_centered_change_bars(cls, ax, chart_data):
        subregions = chart_data["subregions"]
        centers = chart_data["centers"]
        bounds = chart_data["bounds"]
        gdf = chart_data["gdf"]

        if gdf is not None and not gdf.empty:
            gdf.boundary.plot(ax=ax, color="#999", linewidth=0.4, alpha=0.4)

        x_min, y_min, x_max, y_max = bounds
        span_x = max(x_max - x_min, 1e-9)
        span_y = max(y_max - y_min, 1e-9)
        span_min = min(span_x, span_y)

        all_centers = list(centers.values())
        max_abs_net_change = cls._get_max_abs_category_change(subregions)

        for subregion in subregions:
            center = centers.get(subregion["region_id"])
            if center is None:
                continue
            bar_half_width, bar_max_height = cls._get_change_bar_geometry(
                center, all_centers, bounds, span_min
            )
            draw_ctx = {
                "bar_half_width": bar_half_width,
                "bar_max_height": bar_max_height,
                "max_abs_net_change": max_abs_net_change,
                "category_labels": chart_data["category_labels"],
                "category_to_color": chart_data["category_to_color"],
                "span_min": span_min,
            }
            cls._draw_change_bar_for_subregion(
                ax, subregion, center, draw_ctx
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

        category_labels = chart_data["category_labels"]
        category_to_color = chart_data["category_to_color"]
        legend_items = category_labels[: min(len(category_labels), 10)]
        if legend_items:
            value_to_color = {
                label: category_to_color[label]
                for label in legend_items
                if label in category_to_color
            }
            Legend.draw(value_to_color, ax)

    @classmethod
    def _draw_grid_pies(cls, ax, chart_data):
        subregions = chart_data["subregions"]
        category_to_color = chart_data["category_to_color"]
        if not subregions:
            ax.set_axis_off()
            return

        n = len(subregions)
        n_cols = max(1, math.ceil(math.sqrt(n)))
        n_rows = math.ceil(n / n_cols)

        ax.set_axis_off()
        for idx, subregion in enumerate(subregions):
            row = idx // n_cols
            col = idx % n_cols
            left = col / n_cols
            bottom = 1 - (row + 1) / n_rows
            width = 1 / n_cols
            height = 1 / n_rows
            inset = ax.inset_axes(
                [
                    left + 0.02 * width,
                    bottom + 0.05 * height,
                    width * 0.96,
                    height * 0.9,
                ]
            )
            ordered_items = cls._order_positive_values_with_top_first(
                subregion["values"]
            )
            if not ordered_items:
                inset.set_axis_off()
                continue
            labels = [label for label, _ in ordered_items]
            values = [value for _, value in ordered_items]
            colors = [category_to_color[label] for label in labels]
            startangle = cls._get_startangle_with_top_half(
                values[0], sum(values)
            )
            inset.pie(
                values,
                colors=colors,
                startangle=startangle,
                counterclock=False,
                wedgeprops={"linewidth": 0.2, "edgecolor": "white"},
            )
            inset.text(
                0.5,
                -0.08,
                (
                    f"{subregion['region_name']}\n"
                    f"{cls._format_population(subregion['total_value'])}"
                ),
                transform=inset.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                clip_on=False,
            )
            inset.axis("equal")

    # ── main entry point ───────────────────────────────────────────────────

    def draw(self, dataset, fig):
        data_list = dataset.get_data_table()

        subregions = []
        for data in data_list:
            values = data.get("values") or {}
            if not values:
                continue
            values = dict(sorted(values.items(), key=lambda item: -item[1]))
            total_value = data.get("total_value")
            if total_value is None:
                total_value = sum(values.values())
            pct_values = data.get("pct_values") or {
                k: (v / total_value if total_value else 0)
                for k, v in values.items()
            }
            subregions.append(
                {
                    "region_id": data.get("region_id"),
                    "region_name": data.get("region_name")
                    or str(data.get("region_id")),
                    "values": values,
                    "values1": data.get("values1"),
                    "values2": data.get("values2"),
                    "total_value": total_value,
                    "pct_values": pct_values,
                }
            )

        category_total_map = {}
        for subregion in subregions:
            for category, value in subregion["values"].items():
                category_total_map[category] = (
                    category_total_map.get(category, 0) + value
                )
        category_labels = [
            category
            for category, _ in sorted(
                category_total_map.items(), key=lambda item: -item[1]
            )
        ]

        _, value_to_color = ColorSpecFactory.get_color_spec(
            dataset, self.how_cmd
        ).unpack()
        if value_to_color is None:
            value_to_color = {}
        category_to_color = {}
        cmap = plt.get_cmap("tab20")
        n_labels = max(len(category_labels), 1)
        for i, label in enumerate(category_labels):
            color = value_to_color.get(label)
            if color is None:
                for key, key_color in value_to_color.items():
                    if key.startswith(f"{label} ("):
                        color = key_color
                        break
            if color is None:
                color = ColorSpec.LABEL_TO_COLOR.get(label)
            if color is None:
                color = cmap(i / n_labels)
            category_to_color[label] = color

        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])

        if not subregions:
            ax.set_axis_off()
            return

        if self._is_change_chart(subregions):
            subregions = self._sort_subregions(subregions)
            if not category_labels:
                ax.set_axis_off()
                return
            x_values = list(range(len(subregions)))
            y_min, y_max = self._draw_stacked_bars(
                ax, subregions, x_values, category_labels, category_to_color
            )
            self._style_axis(ax, subregions, y_min, y_max, "Population")
            self._add_bar_labels(ax)
            self._draw_category_legend(ax, category_labels, category_to_color)
            return

        chart_data = {
            "subregions": subregions,
            "category_labels": category_labels,
            "category_to_color": category_to_color,
            "centers": None,
            "bounds": None,
            "gdf": None,
        }
        self._draw_grid_pies(ax, chart_data)
