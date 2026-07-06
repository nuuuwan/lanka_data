from lanka_data.visual.plot_visual.BarChartVisual import BarChartVisual


class StackedBarChartVisual(BarChartVisual):
    MIN_LABEL_HEIGHT_PX = 12

    @staticmethod
    def _normalize_subregions(subregions):
        normalized = []
        for subregion in subregions:
            values = subregion["values"]
            total = sum(abs(v) for v in values.values()) or 1
            entry = dict(subregion)
            entry["values"] = {k: v / total for k, v in values.items() if v}
            normalized.append(entry)
        return normalized

    @staticmethod
    def _format_millions(value, _):
        return f"{value:.0%}"

    def _add_share_labels(self, ax, subregions):
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height == 0:
                    continue
                p0 = ax.transData.transform((bar.get_x(), bar.get_y()))
                p1 = ax.transData.transform(
                    (bar.get_x(), bar.get_y() + height)
                )
                if abs(p1[1] - p0[1]) < self.MIN_LABEL_HEIGHT_PX:
                    continue
                fc = bar.get_facecolor()
                lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    f"{height:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#333" if lum > 0.5 else "#eee",
                    clip_on=True,
                )

    def draw(self, dataset, fig):
        subregions = self._build_subregions(dataset.get_data_table())
        category_labels = self._build_category_labels(subregions)
        category_to_color = self._build_category_to_color(
            dataset, category_labels
        )
        subregions = self._normalize_subregions(
            self._sort_subregions(subregions)
        )
        ax = fig.add_subplot(fig.add_gridspec(1, 1)[0])
        if not subregions or not category_labels:
            ax.set_axis_off()
            return
        x_values = list(range(len(subregions)))
        y_min, y_max = self._draw_stacked_bars(
            ax, subregions, x_values, category_labels, category_to_color
        )
        self._style_axis(ax, subregions, y_min, y_max, "Share")
        self._add_share_labels(ax, subregions)
        self._draw_category_legend(ax, category_labels, category_to_color)
