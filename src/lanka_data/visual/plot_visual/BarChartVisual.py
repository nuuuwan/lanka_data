from matplotlib.ticker import FuncFormatter

from lanka_data.visual.plot.Legend import Legend
from lanka_data.visual.plot_visual.PlotVisual import PlotVisual


class BarChartVisual(PlotVisual):
    BOTTOM_PADDING = 0.08

    @staticmethod
    def _is_change_chart(subregions):
        return any(v < 0 for s in subregions for v in s["values"].values())

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

    @staticmethod
    def _draw_stacked_bars(
        ax, subregions, x_values, category_labels, category_to_color
    ):
        y_max = y_min = 0
        for i, subregion in enumerate(subregions):
            values = subregion["values"]
            pos_bottom = neg_bottom = 0
            pos_cats = sorted(
                [c for c in category_labels if values.get(c, 0) >= 0],
                key=lambda c, vm=values: vm.get(c, 0),
            )
            neg_cats = sorted(
                [c for c in category_labels if values.get(c, 0) < 0],
                key=lambda c, vm=values: vm.get(c, 0),
                reverse=True,
            )
            for cat in pos_cats:
                v = values.get(cat, 0)
                if v == 0:
                    continue
                ax.bar(
                    [x_values[i]],
                    [v],
                    bottom=[pos_bottom],
                    color=category_to_color[cat],
                    label=cat,
                    width=0.85,
                )
                pos_bottom += v
            for cat in neg_cats:
                v = values.get(cat, 0)
                ax.bar(
                    [x_values[i]],
                    [v],
                    bottom=[neg_bottom],
                    color=category_to_color[cat],
                    label=cat,
                    width=0.85,
                )
                neg_bottom += v
            y_max = max(y_max, pos_bottom)
            y_min = min(y_min, neg_bottom)
        return y_min, y_max

    def _style_axis(self, ax, subregions, y_min, y_max, y_label="Population"):
        x_labels = [s["region_name"] for s in subregions]
        ax.set_xticks(range(len(subregions)))
        ax.set_xticklabels(
            x_labels,
            rotation=90 if len(x_labels) > 12 else 45,
            ha="right",
            fontsize=8,
        )
        pos = ax.get_position()
        padded_h = max(pos.height - self.BOTTOM_PADDING, pos.height * 0.7)
        ax.set_position(
            [pos.x0, pos.y0 + self.BOTTOM_PADDING, pos.width, padded_h]
        )
        ax.grid(False)
        ax.margins(x=0.06, y=0.12)
        y_pad = max(y_max - y_min, 1) * 0.12
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.axhline(0, color="#666", linewidth=0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(self._format_millions))
        ax.set_ylabel(y_label)

    @staticmethod
    def _draw_category_legend(ax, category_labels, category_to_color):
        items = {
            lbl: category_to_color[lbl]
            for lbl in category_labels[:10]
            if lbl in category_to_color
        }
        if items:
            Legend.draw(items, ax)

    # horizontal text: chars run along bar width, lines stack along bar height
    _CHAR_W_RATIO = 0.6
    _LINE_H_RATIO = 1.4
    _MIN_FONT = 3
    _MAX_FONT = 24

    @classmethod
    def _fit_fontsize(cls, bar_h_px, bar_w_px, n_chars, n_lines, dpi):
        pt_per_px = 72 / dpi
        return min(
            cls._MAX_FONT,
            bar_w_px * pt_per_px / max(n_chars * cls._CHAR_W_RATIO, 1),
            bar_h_px * pt_per_px / max(n_lines * cls._LINE_H_RATIO, 1),
        )

    def _add_bar_labels(self, ax, subregions):
        is_change = self._is_change_chart(subregions)
        totals = {
            i: sum(abs(v) for v in s["values"].values()) or 1
            for i, s in enumerate(subregions)
        }
        for container in ax.containers:
            cat = container.get_label()
            for bar in container:
                height = bar.get_height()
                if height == 0:
                    continue
                p0 = ax.transData.transform((bar.get_x(), bar.get_y()))
                p1 = ax.transData.transform(
                    (bar.get_x() + bar.get_width(), bar.get_y() + height)
                )
                bar_h_px = abs(p1[1] - p0[1])
                bar_w_px = abs(p1[0] - p0[0])
                idx = round(bar.get_x() + bar.get_width() / 2)
                abs_label = self._format_millions(height, None)
                if is_change:
                    pct_val = subregions[idx]["pct_values"].get(cat, 0)
                    pct_label = f"{pct_val * 100:+.1f}pp"
                else:
                    total = totals.get(idx, 1)
                    pct_label = f"{abs(height) / total:.1%}"
                dpi = ax.get_figure().dpi
                full_text = f"{abs_label}\n{pct_label}"
                max_line = max(len(abs_label), len(pct_label))
                fontsize = self._fit_fontsize(
                    bar_h_px, bar_w_px, max_line, 2, dpi
                )
                if fontsize < self._MIN_FONT:
                    # try pct only (1 line)
                    fontsize = self._fit_fontsize(
                        bar_h_px, bar_w_px, len(pct_label), 1, dpi
                    )
                    if fontsize < self._MIN_FONT:
                        continue
                    text = pct_label
                else:
                    text = full_text
                fc = bar.get_facecolor()
                lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="#333" if lum > 0.5 else "#eee",
                    clip_on=True,
                )

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

        x_values = list(range(len(subregions)))
        y_min, y_max = self._draw_stacked_bars(
            ax, subregions, x_values, category_labels, category_to_color
        )
        self._style_axis(ax, subregions, y_min, y_max, "Population")
        self._add_bar_labels(ax, subregions)
        self._draw_category_legend(ax, category_labels, category_to_color)
