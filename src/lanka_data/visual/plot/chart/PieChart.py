import math

from matplotlib.patches import Rectangle, Wedge

from lanka_data.visual.plot.chart.AbstractChart import AbstractChart
from lanka_data.visual.plot.chart.BarChart import BarChart
from lanka_data.visual.plot.Legend import Legend


class PieChart(AbstractChart):
    CHART_TYPE = "PieChart"

    @staticmethod
    def _is_change_chart(subregions):
        return any(
            value < 0
            for subregion in subregions
            for value in subregion["values"].values()
        )

    @staticmethod
    def _is_change_with_both_signs(subregions):
        has_positive = False
        has_negative = False
        for subregion in subregions:
            for value in subregion["values"].values():
                if value > 0:
                    has_positive = True
                elif value < 0:
                    has_negative = True
        return has_positive and has_negative

    @staticmethod
    def _format_signed_population(total_value):
        label = ""
        if total_value is not None:
            sign = "+" if total_value > 0 else ""
            abs_value = abs(total_value)
            if abs_value >= 1_000_000:
                label = f"{sign}{total_value / 1_000_000:.2f}M"
            elif abs_value >= 1_000:
                label = f"{sign}{total_value / 1_000:.1f}K"
            else:
                label = f"{sign}{total_value:.0f}"
        return label

    @staticmethod
    def _get_max_abs_category_change(subregions):
        max_abs_change = 0
        for subregion in subregions:
            for value in subregion["values"].values():
                max_abs_change = max(max_abs_change, abs(value))
        return max(max_abs_change, 1)

    @staticmethod
    def _get_change_bar_geometry(
        center,
        all_centers,
        bounds,
        span_min,
    ):
        pair_cap = PieChart._pair_radius_cap(
            center,
            all_centers,
            span_min * 0.1,
        )
        bound_cap = PieChart._distance_to_bounds(center, bounds)
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
    def _draw_change_bar_for_subregion(
        cls,
        ax,
        subregion,
        center,
        draw_ctx,
    ):
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

    @staticmethod
    def _draw_pie_at(ax, center, radius, values_map, color_map):
        ordered_items = PieChart._order_positive_values_with_top_first(
            values_map
        )
        total_value = sum(value for _, value in ordered_items)
        if total_value <= 0:
            return

        top_value = ordered_items[0][1]
        theta = PieChart._get_startangle_with_top_half(
            top_value,
            total_value,
        )
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
                ax,
                center,
                radius,
                subregion["values"],
                category_to_color,
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
            Legend.draw(
                value_to_color,
                ax,
            )

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
                center,
                all_centers,
                bounds,
                span_min,
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
                ax,
                subregion,
                center,
                draw_ctx,
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
            Legend.draw(
                value_to_color,
                ax,
            )

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
                values[0],
                sum(values),
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

    def draw_axis(self, ax, chart_data):
        subregions = chart_data["subregions"]
        if not subregions:
            ax.set_axis_off()
            return

        if self._is_change_chart(subregions):
            BarChart(self.how_label, self.params).draw_axis(ax, chart_data)
            return

        centers = chart_data["centers"]
        bounds = chart_data["bounds"]
        if centers and bounds:
            self._draw_map_centered_pies(ax, chart_data)
            return

        self._draw_grid_pies(ax, chart_data)
