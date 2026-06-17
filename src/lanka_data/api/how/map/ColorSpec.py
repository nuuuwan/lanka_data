import random
from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class ColorSpec:
    region_to_color: dict[str, str]
    value_to_color: dict[str, str]

    DEFAULT_CMAP_ABS = plt.cm.get_cmap("RdYlGn")
    DEFAULT_CMAP_DIFF = plt.cm.get_cmap("bwr")
    DEFAULT_CMAP_CAT = plt.cm.get_cmap("tab20")

    @staticmethod
    def p_to_color_for_abs(p):
        return ColorSpec.DEFAULT_CMAP_ABS(p)

    @staticmethod
    def p_to_color_for_diff(p):
        return ColorSpec.DEFAULT_CMAP_DIFF(p)

    @staticmethod
    def p_to_color_for_category(p):
        return ColorSpec.DEFAULT_CMAP_CAT(p)

    def unpack(self):
        return self.region_to_color, self.value_to_color

    @classmethod
    def by_custom_key(cls, result_data, func_key_getter, hide_legend):
        data_list = result_data["data_list"]
        sorted_color_keys = sorted(
            list(set([func_key_getter(data) for data in data_list]))
        )
        if hide_legend:
            random.shuffle(sorted_color_keys)

        n_keys = len(sorted_color_keys)
        region_to_color = {}
        value_to_color = {}
        for data in data_list:
            key = func_key_getter(data)
            i_key = sorted_color_keys.index(key)
            p = i_key / (n_keys - 1) if n_keys > 1 else 0
            color = ColorSpec.p_to_color_for_category(p)
            region_id = data["region_id"]
            region_to_color[region_id] = color
            value_to_color[key] = color

        if hide_legend:
            value_to_color = None
        return cls(region_to_color, value_to_color)

    @classmethod
    def by_single_pct_value(cls, result_data, how):
        single_pct_value = how.params
        data_list = result_data["data_list"]
        pct_values = [
            data["pct_values"][single_pct_value] for data in data_list
        ]
        value_to_rank = {v: r for r, v in enumerate(sorted(set(pct_values)))}
        n = len(value_to_rank)
        value_to_color, region_to_color = {}, {}
        for data in data_list:
            value = data["pct_values"][single_pct_value]
            rank = value_to_rank[value]
            color = ColorSpec.p_to_color_for_abs(1 - rank / (n - 1))
            value_to_color[value] = color
            region_to_color[data["region_id"]] = color
        return cls(region_to_color, value_to_color)

    @classmethod
    def by_region_to_custom_value(cls, region_to_custom_value, is_diff):
        sorted_custom_values = list(sorted(region_to_custom_value.values()))
        n = len(sorted_custom_values)
        region_to_color = {}
        value_to_color = {}
        for region_id, custom_value in region_to_custom_value.items():
            i_values = sorted_custom_values.index(custom_value)
            p = i_values / (n - 1) if n > 1 else 0
            if is_diff:
                value = f"{custom_value:+.4f}"
                color = ColorSpec.p_to_color_for_diff(p)
            else:
                value = f"{custom_value:.4f}"
                color = ColorSpec.p_to_color_for_abs(p)

            region_to_color[region_id] = color
            value_to_color[value] = color
        return cls(region_to_color, value_to_color)
