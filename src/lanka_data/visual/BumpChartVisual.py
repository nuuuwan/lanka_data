import matplotlib.pyplot as plt

from lanka_data.visual.plot.chart.BumpChart import BumpChart
from lanka_data.visual.plot.color_spec import ColorSpec, ColorSpecFactory
from lanka_data.visual.PlotVisual import PlotVisual


class BumpChartVisual(PlotVisual):
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

        when_labels = None
        when_cmd = getattr(self.command, "when_cmd", None)
        if when_cmd and "-" in when_cmd:
            tokens = when_cmd.split("-")
            if len(tokens) == 2:
                when_labels = tokens

        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])

        chart_data = {
            "subregions": subregions,
            "category_labels": category_labels,
            "category_to_color": category_to_color,
            "when_labels": when_labels,
            "y_label": "Population",
        }
        BumpChart().draw_axis(ax, chart_data)
