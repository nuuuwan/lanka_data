import math

from matplotlib.patches import Patch, Wedge

from lanka_data.api.how.chart.AbstractChart import AbstractChart


class PieChart(AbstractChart):
    CHART_TYPE = "PieChart"

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
    def _draw_pie_at(ax, center, radius, values_map, color_map):
        total_value = sum(values_map.values())
        if total_value <= 0:
            return

        theta = 90
        for label, value in values_map.items():
            if value <= 0:
                continue
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
            handles = [
                Patch(facecolor=category_to_color[label], label=label)
                for label in legend_items
            ]
            ax.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=7,
                frameon=False,
                ncol=2,
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

            labels = list(subregion["values"].keys())
            values = [subregion["values"][label] for label in labels]
            colors = [category_to_color[label] for label in labels]
            inset.pie(
                values,
                colors=colors,
                startangle=90,
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

        centers = chart_data["centers"]
        bounds = chart_data["bounds"]
        if centers and bounds:
            self._draw_map_centered_pies(ax, chart_data)
            return

        self._draw_grid_pies(ax, chart_data)
