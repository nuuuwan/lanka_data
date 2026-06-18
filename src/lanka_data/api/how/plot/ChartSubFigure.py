import matplotlib.pyplot as plt

from lanka_data.api.how.map.color_spec import ColorSpecFactory
from lanka_data.api.how.plot.Text import Text


class ChartSubFigure:
    def __init__(self, figure_label, command, is_cartogram, subfigure):
        self.figure_label = figure_label
        self.command = command
        self.is_cartogram = is_cartogram
        self.subfigure = subfigure
        gs = self.subfigure.add_gridspec(1, 1)
        self.ax = self.subfigure.add_subplot(gs[0])

    def _get_data_series(self, result_data):
        aggr_data = result_data.get("aggr_data") or {}
        values = aggr_data.get("values") or {}
        pct_values = aggr_data.get("pct_values") or {}

        if values:
            labels = list(values.keys())
            value_list = [values[label] for label in labels]
            pct_list = [pct_values.get(label, 0) for label in labels]
            return labels, value_list, pct_list

        data_list = result_data.get("data_list") or []
        labels = [
            data.get("region_name") or str(data.get("region_id"))
            for data in data_list
        ]
        value_list = [1 for _ in labels]
        n = len(labels)
        pct_list = [1 / n for _ in labels] if n > 0 else []
        return labels, value_list, pct_list

    def _get_color_map_for_labels(self, labels):
        _, value_to_color = ColorSpecFactory.get_color_spec(
            self.command
        ).unpack()
        if value_to_color is None:
            value_to_color = {}

        colors = []
        cmap = plt.get_cmap("tab20")
        n_labels = max(len(labels), 1)
        for i, label in enumerate(labels):
            color = value_to_color.get(label)
            if color is None:
                for key, key_color in value_to_color.items():
                    if key.startswith(f"{label} ("):
                        color = key_color
                        break
            if color is None:
                color = cmap(i / n_labels)
            colors.append(color)
        return colors

    def draw(self):
        how = self.command.get_how()
        what = self.command.get_what()
        when = self.command.get_when()
        where = self.command.get_where()

        result_data = how.get_data(what, when, where)
        labels, values, pct_values = self._get_data_series(result_data)
        colors = self._get_color_map_for_labels(labels)
        how.draw_axis(self.ax, labels, values, pct_values, colors)
        if self.figure_label:
            Text.plot(
                self.subfigure,
                (0.5, 0.9),
                self.figure_label,
                fontsize=16,
                color="#000",
            )
        return result_data
