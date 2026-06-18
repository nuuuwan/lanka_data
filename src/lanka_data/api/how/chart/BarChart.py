from matplotlib.ticker import FuncFormatter

from lanka_data.api.how.chart.AbstractChart import AbstractChart


class BarChart(AbstractChart):
    CHART_TYPE = "BarChart"
    BOTTOM_PADDING = 0.08

    @staticmethod
    def _format_millions(value, _):
        label = "0"
        if value != 0:
            if abs(value) >= 1_000_000:
                label = f"{value / 1_000_000:.1f}M"
            elif abs(value) >= 1_000:
                label = f"{value / 1_000:.0f}K"
            else:
                label = f"{value:.0f}"
        return label

    def draw_axis(self, ax, chart_data):
        subregions = chart_data["subregions"]
        category_labels = chart_data["category_labels"]
        category_to_color = chart_data["category_to_color"]

        if not subregions or not category_labels:
            ax.set_axis_off()
            return

        x_values = list(range(len(subregions)))
        bottoms = [0 for _ in subregions]
        for category in category_labels:
            y_values = [
                subregion["values"].get(category, 0)
                for subregion in subregions
            ]
            ax.bar(
                x_values,
                y_values,
                bottom=bottoms,
                color=category_to_color[category],
                label=category,
                width=0.85,
            )
            bottoms = [
                bottom + value for bottom, value in zip(bottoms, y_values)
            ]

        ax.set_xticks(x_values)
        x_labels = [subregion["region_name"] for subregion in subregions]
        rotation = 90 if len(x_labels) > 12 else 45
        ax.set_xticklabels(
            x_labels,
            rotation=rotation,
            ha="right",
            fontsize=8,
        )
        # Reserve vertical space for rotated labels
        # above the global footer.
        pos = ax.get_position()
        padded_height = max(pos.height - self.BOTTOM_PADDING, pos.height * 0.7)
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
        y_max = max(bottoms) if bottoms else 0
        if y_max > 0:
            ax.set_ylim(0, y_max * 1.12)
        ax.yaxis.set_major_formatter(FuncFormatter(self._format_millions))
        ax.set_ylabel("Population")

        legend_labels = category_labels[: min(len(category_labels), 10)]
        if legend_labels:
            handles, labels = ax.get_legend_handles_labels()
            selected = [
                (h, l)
                for h, l in zip(handles, labels)
                if l in set(legend_labels)
            ]
            if selected:
                sel_handles = [h for h, _ in selected]
                sel_labels = [l for _, l in selected]
                ax.legend(
                    sel_handles,
                    sel_labels,
                    fontsize=7,
                    frameon=False,
                    ncol=2,
                    loc="upper right",
                )
