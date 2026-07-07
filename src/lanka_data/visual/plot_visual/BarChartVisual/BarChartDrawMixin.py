from matplotlib.ticker import FuncFormatter

from lanka_data.visual.plot.Legend import Legend
from lanka_data.visual.plot.Style import Style


class BarChartDrawMixin:
    PADDING = 0.08

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
                    width=0.8,
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
                    width=0.8,
                )
                neg_bottom += v
            y_max = max(y_max, pos_bottom)
            y_min = min(y_min, neg_bottom)
        return y_min, y_max

    def _reposition_axis(self, ax, bottom_padding):
        pos = ax.get_position()
        padded_w = pos.width - self.PADDING * 2
        padded_h = max(
            pos.height - bottom_padding - self.PADDING,
            pos.height * 0.4,
        )
        ax.set_position(
            [
                pos.x0 + self.PADDING,
                pos.y0 + bottom_padding,
                padded_w,
                padded_h,
            ]
        )

    def _style_axis(
        self,
        ax,
        subregions,
        y_min,
        y_max,
        y_label="Population",
        x_labels=True,
    ):
        x_names = [s["region_name"] for s in subregions]
        if x_labels:
            bottom_padding = self._fit_x_labels(ax, x_names)
        else:
            ax.set_xticks([])
            bottom_padding = 0
        self._reposition_axis(ax, bottom_padding)
        ax.grid(
            True, axis="y", color=Style.COLOR_GRID, linewidth=0.5, zorder=-1
        )
        ax.margins(x=0.0, y=0.0)
        ax.set_xlim(-0.6, len(x_names) - 0.4)
        y_top_pad = 0 if x_labels else (y_max - y_min) * 0.08
        ax.set_ylim(y_min, y_max + y_top_pad)
        ax.axhline(0, color=Style.COLOR_AXIS, linewidth=0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(self._format_millions))
        ax.set_ylabel(y_label, color=Style.COLOR_METADATA)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)

    @staticmethod
    def _draw_category_legend(ax, category_labels, category_to_color):
        items = {
            lbl: category_to_color[lbl]
            for lbl in category_labels[:10]
            if lbl in category_to_color
        }
        if items:
            Legend.draw(items, ax)
