from lanka_data.visual.plot_visual.BarChartVisual import BarChartVisual

from .PieChartGridMixin import PieChartGridMixin


class PieChartVisual(PieChartGridMixin, BarChartVisual):
    @staticmethod
    def _format_population(total_value):
        if total_value is None:
            return ""
        if total_value >= 1_000_000:
            return f"{total_value / 1_000_000:.2f}M"
        divisor = 1_000 if total_value >= 1_000 else 1
        fmt = ".1f" if total_value >= 1_000 else ".0f"
        suffix = "K" if total_value >= 1_000 else ""
        return f"{total_value / divisor:{fmt}}{suffix}"

    @staticmethod
    def _order_positive_values_with_top_first(values_map):
        items = [(k, v) for k, v in values_map.items() if v > 0]
        if not items:
            return []
        top_label, _ = max(items, key=lambda x: x[1])
        return [(k, v) for k, v in items if k == top_label] + [
            (k, v) for k, v in items if k != top_label
        ]

    @staticmethod
    def _get_startangle(top_value, total_value):
        if total_value <= 0:
            return 90
        return 90 + 360.0 * top_value / total_value / 2

    def draw(self, dataset, fig):
        subregions = self._build_subregions(dataset.get_data_table())
        category_labels = self._build_category_labels(subregions)
        category_to_color = self._build_category_to_color(
            dataset, category_labels
        )
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
            self._add_bar_labels(ax, subregions)
            self._draw_category_legend(ax, category_labels, category_to_color)
            return
        self._draw_grid_pies(ax, subregions, category_to_color)
