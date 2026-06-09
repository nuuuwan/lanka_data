import matplotlib.pyplot as plt

from lanka_data.how.map.ColorUtils import ColorUtils


class OrderColorUtils:
    _PARAM_IDX = {"Top": 0, "2nd": 1, "3rd": 2, "Bottom": -1}
    DEFAULT_MATPLOTLIB_CMAP = "Set1"

    @staticmethod
    def _func_key_getter(how, what):
        idx = OrderColorUtils._PARAM_IDX.get(how.params or "Top")
        if idx is None:
            return None

        def func_key_getter(data):
            return list(what.get_values(data).keys())[idx]

        return func_key_getter

    @staticmethod
    def _build_key_pcts(data_list, func_key_getter):
        key_to_base_hex = {}
        value_to_color = {}
        all_pcts, raw_pcts = [], {}

        # First pass: collect all unknown keys in order of first appearance
        unknown_keys = []
        for data in data_list:
            key = func_key_getter(data) if func_key_getter else None
            if key not in key_to_base_hex and key not in ColorUtils.HUE_IDX:
                if key not in unknown_keys:
                    unknown_keys.append(key)

        cmap = plt.get_cmap(OrderColorUtils.DEFAULT_MATPLOTLIB_CMAP)
        for data in data_list:
            key = func_key_getter(data) if func_key_getter else None
            if key not in key_to_base_hex:
                if key in ColorUtils.HUE_IDX:
                    key_to_base_hex[key] = ColorUtils.hue_to_hex(
                        ColorUtils.HUE_IDX[key]
                    )
                else:
                    i_key = unknown_keys.index(key)
                    r, g, b, _ = cmap(i_key % 20)
                    key_to_base_hex[key] = ColorUtils.rgb_to_hex((r, g, b))
                value_to_color[key] = ColorUtils._color_with_opacity(
                    key_to_base_hex[key], 1.0
                )
            pct_dict = (
                data.get("pct_values") or data.get("pct_votes_by_party") or {}
            )
            pct = pct_dict.get(key, 0.5)
            all_pcts.append(pct)
            raw_pcts[data["region_id"]] = (key, pct)
        return key_to_base_hex, value_to_color, all_pcts, raw_pcts

    @staticmethod
    def get_order_color_map(result_data, how, what, func_key_getter):
        data_list = result_data["data_list"]
        key_to_base_hex, value_to_color, all_pcts, raw_pcts = (
            OrderColorUtils._build_key_pcts(data_list, func_key_getter)
        )
        pct_min, pct_max = min(all_pcts), max(all_pcts)
        pct_span = pct_max - pct_min if pct_max > pct_min else 1.0
        region_color_map = {}
        for region_id, (key, pct) in raw_pcts.items():
            normalised = (pct - pct_min) / pct_span
            region_color_map[region_id] = ColorUtils._color_with_opacity(
                key_to_base_hex[key], normalised
            )
        cat_pct_ranges = {}
        for key, pct in raw_pcts.values():
            lo, hi = cat_pct_ranges.get(key, (pct, pct))
            cat_pct_ranges[key] = (min(lo, pct), max(hi, pct))
        value_to_color["__pct_range__"] = (pct_min, pct_max)
        value_to_color["__cat_pct_ranges__"] = cat_pct_ranges
        return region_color_map, value_to_color
