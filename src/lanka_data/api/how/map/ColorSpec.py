from dataclasses import dataclass

from lanka_data.api.how.map.ColorUtils import ColorUtils


@dataclass
class ColorSpec:
    region_to_color: dict[str, str]
    value_to_color: dict[str, str]

    @classmethod
    def by_custom_key(result_data, func_key_getter, hide_legend=False):
        data_list = result_data["data_list"]
        sorted_color_keys = sorted(
            list(set([func_key_getter(data) for data in data_list]))
        )

        n_keys = len(sorted_color_keys)
        region_to_color = {}
        value_to_color = {}
        for data in data_list:
            key = func_key_getter(data)
            i_key = sorted_color_keys.index(key)
            p = i_key / (n_keys - 1) if n_keys > 1 else 0
            color = ColorUtils.p_to_color_for_category(p)
            region_id = data["region_id"]
            region_to_color[region_id] = color
            value_to_color[key] = color

        if hide_legend:
            value_to_color = None
        return cls(region_to_color, value_to_color)

    @classmethod
    def by_single_pct_value(result_data, how):
        single_pct_value = how.params
        data_list = result_data["data_list"]
        pct_values = [
            data["pct_values"][single_pct_value] for data in data_list
        ]
        value_to_rank = {v: r for r, v in enumerate(sorted(set(pct_values)))}
        n = len(value_to_rank)
        value_to_color, region_color_map = {}, {}
        for data in data_list:
            value = data["pct_values"][single_pct_value]
            rank = value_to_rank[value]
            color = ColorUtils.p_to_color_for_abs(1 - rank / (n - 1))
            value_to_color[value] = color
            region_color_map[data["region_id"]] = color
        return cls(region_color_map, value_to_color)

    @classmethod
    def by_region_to_custom_value(cls, region_to_custom_value, is_diff=False):
        sorted_custom_values = list(sorted(region_to_custom_value.values()))
        n = len(sorted_custom_values)
        region_to_color = {}
        value_to_color = {}
        for region_id, custom_value in region_to_custom_value.items():
            i_values = sorted_custom_values.index(custom_value)
            p = i_values / (n - 1) if n > 1 else 0
            if is_diff:
                value = f"{custom_value:+.4f}"
                color = ColorUtils.p_to_color_for_diff(p)
            else:
                value = f"{custom_value:.4f}"
                color = ColorUtils.p_to_color_for_abs(p)

            region_to_color[region_id] = color
            value_to_color[value] = color
        return cls(region_to_color, value_to_color)
