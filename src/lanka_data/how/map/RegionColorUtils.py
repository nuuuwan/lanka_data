import colorsys

import matplotlib.pyplot as plt

from lanka_data.how.map.OrderColorUtils import OrderColorUtils


class RegionColorUtils:
    @staticmethod
    def _colors_no_values(result_data):
        data_list = result_data["data_list"]
        cmap = plt.get_cmap(OrderColorUtils.DEFAULT_MATPLOTLIB_CMAP)
        return (
            {
                data["region_id"]: cmap(i % 20)
                for i, data in enumerate(data_list)
            },
            None,
        )

    @staticmethod
    def _colors_values_key(result_data, how, what):
        data_list = result_data["data_list"]
        pct_values = [data["pct_values"][how.params] for data in data_list]
        value_to_rank = {v: r for r, v in enumerate(sorted(set(pct_values)))}
        n = len(value_to_rank)
        value_to_color, region_color_map = {}, {}
        for data in data_list:
            value = data["pct_values"][how.params]
            rank = value_to_rank[value]
            color = colorsys.hls_to_rgb((1 - rank / (n - 1)) * 0.67, 0.5, 1.0)
            value_to_color[value] = color
            region_color_map[data["region_id"]] = color
        return region_color_map, value_to_color

    @staticmethod
    def _colors_with_values(result_data, how, what):
        func_key_getter = OrderColorUtils._func_key_getter(how, what)
        if func_key_getter:
            return OrderColorUtils.get_order_color_map(
                result_data, how, what, func_key_getter
            )
        return RegionColorUtils._colors_values_key(result_data, how, what)

    @staticmethod
    def get_region_color_map(result_data, how, what):
        data_list = result_data["data_list"]
        if what.get_values(data_list[0]) is None:
            return RegionColorUtils._colors_no_values(result_data)
        return RegionColorUtils._colors_with_values(result_data, how, what)
