from lanka_data.api.how.chart.AbstractChart import AbstractChart


class PieChart(AbstractChart):
    CHART_TYPE = "PieChart"

    def draw_axis(self, ax, labels, values, pct_values, colors):
        if not values:
            ax.set_axis_off()
            return

        ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            textprops={"fontsize": 10},
        )
        ax.axis("equal")
