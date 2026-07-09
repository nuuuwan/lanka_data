from lanka_data.visual.formatters.UnitFormatter import UnitFormatter
from lanka_data.visual.formatters.WhatFormatter import WhatFormatter


class BarChartLabelMixin:
    def _y_axis_label(self):
        command = getattr(self, "command", None)
        what_cmd = getattr(command, "what_cmd", None)
        label = WhatFormatter(what_cmd).format() if what_cmd else None
        return label or "Population"

    @staticmethod
    def _is_single_category(subregions):
        cats = {
            cat
            for s in subregions
            for cat, v in s["values"].items()
            if v != 0
        }
        return len(cats) == 1

    def _value_label(self, cat, height):
        return UnitFormatter(cat).format(self._format_millions(height, None))

    @staticmethod
    def _compute_pct_label(subregions, is_change, totals, cat, bar):
        height = bar.get_height()
        idx = round(bar.get_x() + bar.get_width() / 2)
        if is_change:
            pct_val = subregions[idx]["pct_values"].get(cat, 0)
            return f"{pct_val * 100:+.1f}pp"
        total = totals.get(idx, 1)
        return f"{abs(height) / total:.1%}"

    def _bar_label_text(
        self, subregions, single, is_change, totals, cat, bar
    ):
        abs_label = self._value_label(cat, bar.get_height())
        if single:
            return abs_label, None
        pct_label = self._compute_pct_label(
            subregions, is_change, totals, cat, bar
        )
        return abs_label, pct_label

    def _add_bar_labels(self, ax, subregions):
        is_change = self._is_change_chart(subregions)
        single = self._is_single_category(subregions)
        totals = {
            i: sum(abs(v) for v in s["values"].values()) or 1
            for i, s in enumerate(subregions)
        }
        dpi = ax.get_figure().dpi
        for container in ax.containers:
            cat = container.get_label()
            for bar in container:
                height = bar.get_height()
                if height == 0:
                    continue
                bar_h_px, bar_w_px = self._bar_pixel_size(ax, bar, height)
                abs_label, pct_label = self._bar_label_text(
                    subregions, single, is_change, totals, cat, bar
                )
                text, fontsize = self._resolve_bar_label(
                    bar_h_px, bar_w_px, abs_label, pct_label, dpi
                )
                self._draw_bar_text(ax, bar, height, text, fontsize)
