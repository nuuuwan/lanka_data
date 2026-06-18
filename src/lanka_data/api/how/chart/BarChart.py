from matplotlib.ticker import FuncFormatter

from lanka_data.api.how.chart.AbstractChart import AbstractChart


class BarChart(AbstractChart):
    CHART_TYPE = "BarChart"
    BOTTOM_PADDING = 0.08

    @staticmethod
    def _is_change_chart(subregions):
        return any(
            value < 0
            for subregion in subregions
            for value in subregion["values"].values()
        )

    @staticmethod
    def _get_net_change_values(subregions):
        return [sum(subregion["values"].values()) for subregion in subregions]

    @classmethod
    def _sort_subregions(cls, subregions):
        is_change_chart = cls._is_change_chart(subregions)
        if is_change_chart:
            return sorted(
                subregions,
                key=lambda subregion: sum(subregion["values"].values()),
                reverse=True,
            )
        return sorted(
            subregions,
            key=lambda subregion: subregion.get(
                "total_value", sum(subregion["values"].values())
            ),
            reverse=True,
        )

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

    @staticmethod
    def _draw_change_grouped_bars(
        ax,
        subregions,
        x_values,
        category_labels,
        category_to_color,
    ):
        n_categories = max(len(category_labels), 1)
        slot_width = 0.84
        bar_width = slot_width / n_categories
        y_max = 0
        y_min = 0

        for category_index, category in enumerate(category_labels):
            offset = (
                -slot_width / 2 + bar_width * category_index + bar_width / 2
            )
            y_values = [
                subregion["values"].get(category, 0)
                for subregion in subregions
            ]
            x_positions = [x + offset for x in x_values]
            ax.bar(
                x_positions,
                y_values,
                color=category_to_color[category],
                label=category,
                width=bar_width * 0.92,
            )
            if y_values:
                y_max = max([y_max] + y_values)
                y_min = min([y_min] + y_values)

        return y_min, y_max

    @staticmethod
    def _draw_stacked_bars(
        ax,
        subregions,
        x_values,
        category_labels,
        category_to_color,
    ):
        pos_bottoms = [0 for _ in subregions]
        neg_bottoms = [0 for _ in subregions]
        for category in category_labels:
            y_values = [
                subregion["values"].get(category, 0)
                for subregion in subregions
            ]
            bar_bottoms = []
            for i, value in enumerate(y_values):
                if value >= 0:
                    bar_bottoms.append(pos_bottoms[i])
                    pos_bottoms[i] += value
                else:
                    bar_bottoms.append(neg_bottoms[i])
                    neg_bottoms[i] += value

            ax.bar(
                x_values,
                y_values,
                bottom=bar_bottoms,
                color=category_to_color[category],
                label=category,
                width=0.85,
            )
        y_max = max(pos_bottoms) if pos_bottoms else 0
        y_min = min(neg_bottoms) if neg_bottoms else 0
        return y_min, y_max

    def _style_axis(self, ax, subregions, y_min, y_max):
        x_values = list(range(len(subregions)))
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
        padded_height = max(
            pos.height - self.BOTTOM_PADDING, pos.height * 0.7
        )
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
        y_span = y_max - y_min
        if y_span <= 0:
            y_span = 1
        y_pad = y_span * 0.12
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.axhline(0, color="#666", linewidth=0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(self._format_millions))
        ax.set_ylabel("Population")

    @staticmethod
    def _draw_category_legend(ax, category_labels):
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

    def draw_axis(self, ax, chart_data):
        subregions = self._sort_subregions(chart_data["subregions"])
        category_labels = chart_data["category_labels"]
        category_to_color = chart_data["category_to_color"]
        is_change_chart = self._is_change_chart(subregions)

        if not subregions or not category_labels:
            ax.set_axis_off()
            return

        x_values = list(range(len(subregions)))
        if is_change_chart:
            y_min, y_max = self._draw_change_grouped_bars(
                ax,
                subregions,
                x_values,
                category_labels,
                category_to_color,
            )
        else:
            y_min, y_max = self._draw_stacked_bars(
                ax,
                subregions,
                x_values,
                category_labels,
                category_to_color,
            )

        self._style_axis(ax, subregions, y_min, y_max)

        self._draw_category_legend(ax, category_labels)
