from lanka_data.visual.plot.Style import Style


class BarChartSingleMixin:
    @staticmethod
    def _single_categories(values, category_labels):
        return [c for c in category_labels if values.get(c, 0) != 0]

    def _draw_single_region(
        self, ax, subregion, category_labels, category_to_color
    ):
        values = subregion["values"]
        cats = self._single_categories(values, category_labels)
        y_max = y_min = 0
        for i, cat in enumerate(cats):
            v = values.get(cat, 0)
            ax.bar(
                [i], [v], color=category_to_color[cat], width=0.85, label=cat
            )
            y_max = max(y_max, v)
            y_min = min(y_min, v)
        pseudo = [{"region_name": c} for c in cats]
        self._style_axis(ax, pseudo, y_min, y_max, self._y_axis_label())
        self._add_single_labels(ax, cats, values)

    def _add_single_labels(self, ax, cats, values):
        for i, cat in enumerate(cats):
            v = values.get(cat, 0)
            ax.text(
                i,
                v,
                self._format_millions(v, None),
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=Style.FONT_SIZE_METADATA,
                color=Style.COLOR_METADATA,
            )
