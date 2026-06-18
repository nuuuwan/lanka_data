import matplotlib.pyplot as plt

from lanka_data.api.how.map.color_spec import ColorSpecFactory
from lanka_data.api.how.map.GeoData import GeoData
from lanka_data.api.how.plot.Text import Text


class ChartSubFigure:
    def __init__(self, figure_label, command, is_cartogram, subfigure):
        self.figure_label = figure_label
        self.command = command
        self.is_cartogram = is_cartogram
        self.subfigure = subfigure
        gs = self.subfigure.add_gridspec(1, 1)
        self.ax = self.subfigure.add_subplot(gs[0])

    @staticmethod
    def _normalize_values(values):
        return dict(sorted(values.items(), key=lambda item: -item[1]))

    def _build_subregion_series(self, result_data):
        data_list = result_data.get("data_list") or []
        subregions = []
        for data in data_list:
            values = data.get("values") or {}
            if not values:
                continue

            values = self._normalize_values(values)
            total_value = data.get("total_value")
            if total_value is None:
                total_value = sum(values.values())
            pct_values = data.get("pct_values")
            if pct_values is None:
                pct_values = {
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

        if subregions:
            return subregions

        aggr_data = result_data.get("aggr_data") or {}
        aggr_values = aggr_data.get("values") or {}
        if not aggr_values:
            return []

        aggr_values = self._normalize_values(aggr_values)
        total_value = aggr_data.get("total_value")
        if total_value is None:
            total_value = sum(aggr_values.values())
        pct_values = aggr_data.get("pct_values") or {
            k: (v / total_value if total_value else 0)
            for k, v in aggr_values.items()
        }

        return [
            {
                "region_id": "Total",
                "region_name": "Total",
                "values": aggr_values,
                "total_value": total_value,
                "pct_values": pct_values,
            }
        ]

    def _get_category_to_color(self, category_labels):
        _, value_to_color = ColorSpecFactory.get_color_spec(
            self.command
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
                color = cmap(i / n_labels)
            category_to_color[label] = color
        return category_to_color

    def _get_geo_context(self, result_data):
        data_list = result_data.get("data_list") or []
        empty_context = dict(gdf=None, centers={}, bounds=None)
        if not data_list:
            return empty_context

        gdf = None
        try:
            gdf = GeoData.get_geopandas_dataframe(
                data_list, self.is_cartogram
            ).copy()
        except Exception:
            gdf = None

        if gdf is not None and not gdf.empty and "geometry" in gdf:
            gdf = gdf[gdf.geometry.notnull()].copy()

        if gdf is None or gdf.empty:
            return empty_context

        centers = {}
        centroids = gdf.geometry.centroid
        for (_, row), centroid in zip(gdf.iterrows(), centroids):
            centers[row["region_id"]] = (float(centroid.x), float(centroid.y))

        bounds = tuple(float(x) for x in gdf.total_bounds)
        return dict(gdf=gdf, centers=centers, bounds=bounds)

    def _build_chart_data(self, result_data):
        subregions = self._build_subregion_series(result_data)

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
        category_to_color = self._get_category_to_color(category_labels)
        geo_context = self._get_geo_context(result_data)
        when_labels = None
        if "-" in self.command.when_cmd:
            tokens = self.command.when_cmd.split("-")
            if len(tokens) == 2:
                when_labels = tokens

        return {
            "subregions": subregions,
            "category_labels": category_labels,
            "category_to_color": category_to_color,
            "gdf": geo_context["gdf"],
            "centers": geo_context["centers"],
            "bounds": geo_context["bounds"],
            "when_labels": when_labels,
        }

    def draw(self):
        how = self.command.get_how()
        what = self.command.get_what()
        when = self.command.get_when()
        where = self.command.get_where()

        result_data = how.get_data(what, when, where)
        chart_data = self._build_chart_data(result_data)
        how.draw_axis(self.ax, chart_data)
        if self.figure_label:
            Text.plot(
                self.subfigure,
                (0.5, 0.9),
                self.figure_label,
                fontsize=16,
                color="#000",
            )
        return result_data
