from lanka_data.api.how.chart.AbstractChart import AbstractChart


class BarChart(AbstractChart):
    CHART_TYPE = "BarChart"

    def draw_axis(self, ax, labels, values, pct_values, colors):
        if not values:
            ax.set_axis_off()
            return

        x_values = list(range(len(labels)))
        ax.bar(x_values, values, color=colors)
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.margins(x=0.02)
