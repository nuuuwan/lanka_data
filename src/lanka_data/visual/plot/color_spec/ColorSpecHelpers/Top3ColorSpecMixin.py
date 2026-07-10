import random

from lanka_data.visual.plot.color_spec.ColorSpec.ColorSpec import ColorSpec


class Top3ColorSpecMixin:
    N_TOP = 3

    @classmethod
    def _top3_label(cls, data):
        keys = list(data.get("pct_values", {}).keys())[: cls.N_TOP]
        if not keys:
            return "(No Data)"
        return "-".join(keys)

    @classmethod
    def get_color_spec_for_top3(cls, dataset) -> ColorSpec:
        data_list = dataset.get_data_table()
        labels = [cls._top3_label(data) for data in data_list]
        sorted_labels = sorted(set(labels))
        random.seed(0)
        random.shuffle(sorted_labels)
        n = len(sorted_labels)
        region_to_color, value_to_color = {}, {}
        for data, label in zip(data_list, labels):
            color = ColorSpec._get_category_color(label, sorted_labels, n)
            region_to_color[data["region_id"]] = color
            value_to_color[label] = color
        return ColorSpec(region_to_color, value_to_color)
