import os
from abc import abstractmethod

import matplotlib.pyplot as plt

from lanka_data.visual.plot import Plot
from lanka_data.visual.plot.color_spec import ColorSpec, ColorSpecFactory
from lanka_data.visual.Visual import Visual


class PlotVisual(Visual):

    @abstractmethod
    def draw(self, dataset, fig):
        pass

    def build(self):
        result = Plot(self).draw()
        image_path = result["image_path"]
        os.system(f"open {image_path}")
        return result

    @staticmethod
    def _build_subregions(data_list):
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
        return subregions

    @staticmethod
    def _build_category_labels(subregions):
        category_total_map = {}
        for subregion in subregions:
            for category, value in subregion["values"].items():
                category_total_map[category] = (
                    category_total_map.get(category, 0) + value
                )
        return [
            cat
            for cat, _ in sorted(
                category_total_map.items(), key=lambda item: -item[1]
            )
        ]

    def _build_category_to_color(self, dataset, category_labels):
        _, value_to_color = ColorSpecFactory.get_color_spec(
            dataset, self.how_cmd
        ).unpack()
        if value_to_color is None:
            value_to_color = {}
        cmap = plt.get_cmap("tab20")
        n_labels = max(len(category_labels), 1)
        category_to_color = {}
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
        return category_to_color
