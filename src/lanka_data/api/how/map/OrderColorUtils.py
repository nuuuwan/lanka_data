import hashlib

from matplotlib.colors import ListedColormap

from lanka_data.api.how.map.ColorUtils import ColorUtils


class OrderColorUtils:
    _PARAM_TO_IDX = {"Top": 0, "2nd": 1, "3rd": 2, "Bottom": -1}
    DEFAULT_CMAP_N_COLORS = 30
    DEFAULT_CMAP = ColorUtils.DEFAULT_CMAP

    @staticmethod
    def generate_cmap():
        n_colors = OrderColorUtils.DEFAULT_CMAP_N_COLORS
        colors = [
            ColorUtils.p_to_color(i / n_colors) for i in range(n_colors)
        ]
        cmap = ListedColormap(colors)
        return cmap

    @staticmethod
    def get_color_for_label(label):
        hash = hashlib.md5(label.encode()).hexdigest()
        hash_int = int(hash, 16) // 999
        i_color = hash_int % OrderColorUtils.DEFAULT_CMAP_N_COLORS
        p = i_color / OrderColorUtils.DEFAULT_CMAP_N_COLORS
        color = ColorUtils.p_to_color(p)
        hex_color = ColorUtils.rgb_to_hex(color[:3])
        return hex_color

    @staticmethod
    def _func_key_getter(how, what):
        idx = OrderColorUtils._PARAM_TO_IDX.get(how.params or "Top")
        if idx is None:
            return None

        def func_key_getter(data):
            values = list(what.get_pct_values(data).keys())
            return values[idx] if idx < len(values) else '(No Data)'

        return func_key_getter

    @staticmethod
    def get_order_color_map(result_data, how, what, func_key_getter):
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
            color = ColorUtils.p_to_color(p)
            region_id = data["region_id"]
            region_to_color[region_id] = color
            value_to_color[key] = color

        return region_to_color, value_to_color
