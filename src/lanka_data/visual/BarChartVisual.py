import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from lanka_data.visual.plot.color_spec import ColorSpec, ColorSpecFactory
from lanka_data.visual.plot.Legend import Legend
from lanka_data.visual.PlotVisual import PlotVisual


class BarChartVisual(PlotVisual):
    BOTTOM_PADDING = 0.08

    @staticmethod
    def _is_change_chart(subregions):
        return any(
            value < 0
            for subregion in subregions
            for value in subregion["values"].values()
        )

    @classmethod
    def _sort_subregions(cls, subregions):
        if cls._is_change_chart(subregions):
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
            elif abs(value) >= 10:
                label = f"{value:.0f}"
            else:
                label = f"{value:.2f}"
        return label

    @staticmethod
    def _draw_stacked_bars(
        ax,
        subregions,
        x_values,
        category_labels,
        category_to_color,
    ):
        y_max = 0
        y_min = 0
        for i, subregion in enumerate(subregions):
            x_value = x_values[i]
            values = subregion["values"]
            pos_bottom = 0
            neg_bottom = 0

            positive_categories = sorted(
                [
                    category
                    for category in category_labels
                    if values.get(category, 0) >= 0
                ],
                key=lambda category, value_map=values: value_map.get(
                    category, 0
                ),
            )
            negative_categories = sorted(
                [
                    category
                    for category in category_labels
                    if values.get(category, 0) < 0
                ],
                key=lambda category, value_map=values: value_map.get(
                    category, 0
                ),
                reverse=True,
            )

            for category in positive_categories:
                value = values.get(category, 0)
                if value == 0:
                    continue
                ax.bar(
                    [x_value],
                    [value],
                    bottom=[pos_bottom],
                    color=category_to_color[category],
                    label=category,
                    width=0.85,
                )
                pos_bottom += value

            for category in negative_categories:
                value = values.get(category, 0)
                ax.bar(
                    [x_value],
                    [value],
                    bottom=[neg_bottom],
                    color=category_to_color[category],
                    label=category,
                    width=0.85,
                )
                neg_bottom += value

            y_max = max(y_max, pos_bottom)
            y_min = min(y_min, neg_bottom)
        return y_min, y_max

    def _style_axis(self, ax, subregions, y_min, y_max, y_label="Population"):
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
        ax.set_ylabel(y_label)

    @staticmethod
    def _draw_category_legend(ax, category_labels, category_to_color):
        legend_labels = category_labels[: min(len(category_labels), 10)]
        if legend_labels:
            value_to_color = {
                label: category_to_color[label]
                for label in legend_labels
                if label in category_to_color
            }
            if value_to_color:
                Legend.draw(value_to_color, ax)

    def _add_bar_labels(self, ax):
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height == 0:
                    continue
                p0 = ax.transData.transform((bar.get_x(), bar.get_y()))
                p1 = ax.transData.transform(
                    (
                        bar.get_x() + bar.get_width(),
                        bar.get_y() + height,
                    )
                )
                bar_h_px = abs(p1[1] - p0[1])
                bar_w_px = abs(p1[0] - p0[0])
                text = self._format_millions(height, None)
                n_chars = max(len(text), 1)
                bar_min_px = min(bar_w_px, bar_h_px)
                fontsize = min(
                    9,
                    bar_min_px * 0.6,
                    bar_h_px / (n_chars * 1.2),
                )
                if fontsize < 3:
                    continue
                x_c = bar.get_x() + bar.get_width() / 2
                y_c = bar.get_y() + height / 2
                fc = bar.get_facecolor()
                r, g, b = fc[0], fc[1], fc[2]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "#333" if lum > 0.5 else "#eee"
                ax.text(
                    x_c,
                    y_c,
                    text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=text_color,
                    rotation=90,
                    clip_on=True,
                )

    def draw(self, dataset, fig):
        data_list = dataset.get_data_table()

        subregions = []
        for data in data_list:
            values = data.get("values") or {}
            if not values:
                continue
            values = dict(sorted(values.items(), key=lambda item: -item[1]))
            total_value = data.get("total_value")
            if total_value is None:
                total_value = sum(values.values())
            pct_values = data.get("pct_values") or {
                k: (v / total_value if total_value else 0)
                for k, v in values.items()
            }
            subregions.append(
                {
                    "region_id": data.get("region_id"),
                    "region_name": data.get("region_name")
                    or str(data.get("region_id")),
                    "values": values,
                    "values1": data.get("values1"),
                    "values2": data.get("values2"),
                    "total_value": total_value,
                    "pct_values": pct_values,
                }
            )

        category_total_map = {}
        for subregion in subregions:
            for category, value in subregion["values"].items():
                category_total_map[category] = (
                    category_total_map.get(category, 0) + value
                )
        category_labels = [
            category
            for category, _ in sorted(
                category_total_map.items(), key=lambda item: -item[1]
            )
        ]

        _, value_to_color = ColorSpecFactory.get_color_spec(
            dataset, self.how_cmd
        ).unpack()
        if value_to_color is None:
            value_to_color = {}
        category_to_color = {}
        cmap = plt.get_cmap("tab20")
        n_labels = max(len(category_labels), 1)
        for i, label in enumerate(category_labels):
            color = value_to_color.get(label)
            if color is None:
                for key, key_color in value_to_color.items():
                    if key.startswith(f"{label} ("):
                        color = key_color
                        break
            if color is None:
                color = ColorSpec.LABEL_TO_COLOR.get(label)
            if color is None:
                color = cmap(i / n_labels)
            category_to_color[label] = color

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
        self._add_bar_labels(ax)
        self._draw_category_legend(ax, category_labels, category_to_color)
