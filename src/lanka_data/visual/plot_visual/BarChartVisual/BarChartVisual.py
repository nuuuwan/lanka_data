from lanka_data.visual.formatters.WhatFormatter import WhatFormatter
from lanka_data.visual.plot_visual.PlotVisual import PlotVisual

from .BarChartDrawMixin import BarChartDrawMixin
from .BarChartLabelMixin import BarChartLabelMixin
from .BarChartSingleMixin import BarChartSingleMixin
from .BarChartXLabelMixin import BarChartXLabelMixin


class BarChartVisual(
    BarChartDrawMixin,
    BarChartLabelMixin,
    BarChartSingleMixin,
    BarChartXLabelMixin,
    PlotVisual,
):
    @staticmethod
    def _is_change_chart(subregions):
        return any(v < 0 for s in subregions for v in s["values"].values())

    def _y_axis_label(self):
        command = getattr(self, "command", None)
        what_cmd = getattr(command, "what_cmd", None)
        label = WhatFormatter(what_cmd).format() if what_cmd else None
        return label or "Population"

    @classmethod
    def _sort_subregions(cls, subregions):
        if cls._is_change_chart(subregions):
            return sorted(
                subregions,
                key=lambda s: sum(s["values"].values()),
                reverse=True,
            )
        return sorted(
            subregions,
            key=lambda s: s.get("total_value", sum(s["values"].values())),
            reverse=True,
        )

    @staticmethod
    def _format_millions(value, _):
        if value == 0:
            return "0"
        av = abs(value)
        if av >= 1_000_000:
            fmt, divisor, suffix = ".1f", 1_000_000, "M"
        elif av >= 1_000:
            fmt, divisor, suffix = ".0f", 1_000, "K"
        else:
            fmt, divisor, suffix = (".0f" if av >= 10 else ".2f"), 1, ""
        return f"{value / divisor:{fmt}}{suffix}"

    def draw(self, dataset, fig):
        subregions = self._build_subregions(dataset.get_data_table())
        category_labels = self._build_category_labels(subregions)
        category_to_color = self._build_category_to_color(
            dataset, category_labels
        )
        subregions = self._sort_subregions(subregions)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        if not subregions or not category_labels:
            ax.set_axis_off()
            return
        if len(subregions) == 1:
            self._draw_single_region(
                ax, subregions[0], category_labels, category_to_color
            )
            return
        x_values = list(range(len(subregions)))
        y_min, y_max = self._draw_stacked_bars(
            ax, subregions, x_values, category_labels, category_to_color
        )
        self._style_axis(
            ax, subregions, y_min, y_max, self._y_axis_label(), x_labels=False
        )
        self._add_bar_labels(ax, subregions)
        self._add_region_labels(ax, subregions, x_values)
        self._draw_category_legend(ax, category_labels, category_to_color)
