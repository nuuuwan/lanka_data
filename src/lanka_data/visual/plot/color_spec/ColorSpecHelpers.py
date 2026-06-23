from lanka_data.data import Diversity, Segregation
from lanka_data.visual.plot.color_spec import ColorSpec


class ColorSpecHelpers:
    KEY_PARAM_TO_I_RANK = {"1st": 0, "2nd": 1, "3rd": 2, "Bottom": -1}
    PCT_VALUE_PARAM_TO_KEY = {
        "1stPct": 0,
        "2ndPct": 1,
        "3rdPct": 2,
    }

    @staticmethod
    def func_key_from_rank(i_rank):
        def f(data):
            pct_values = data["pct_values"]
            keys = list(pct_values.keys())
            return keys[i_rank] if i_rank < len(keys) else "(No Data)"

        return f

    @staticmethod
    def func_pct_value_from_rank(i_rank):
        def f(data):
            pct_values = data["pct_values"]
            values = list(pct_values.values())
            return values[i_rank]

        return f

    @staticmethod
    def func_pct_value_from_key(key):
        def f(data):
            pct_values = data["pct_values"]
            return pct_values[key]

        return f

    @staticmethod
    def get_color_spec_generic(dataset, how_cmd) -> ColorSpec:

        params = how_cmd.split(":")[1] if ":" in how_cmd else "1st"

        if params in ColorSpecHelpers.KEY_PARAM_TO_I_RANK:
            i_rank = ColorSpecHelpers.KEY_PARAM_TO_I_RANK[params]
            func_key = ColorSpecHelpers.func_key_from_rank(i_rank)
            return ColorSpec.by_custom_category_key(
                dataset, func_key, hide_legend=False
            )

        if params in ColorSpecHelpers.PCT_VALUE_PARAM_TO_KEY:
            key = ColorSpecHelpers.PCT_VALUE_PARAM_TO_KEY[params]
            func_value = ColorSpecHelpers.func_pct_value_from_rank(key)
            return ColorSpec.by_single_pct_value(dataset, func_value)

        if params:
            func_value = ColorSpecHelpers.func_pct_value_from_key(params)
            return ColorSpec.by_single_pct_value(
                dataset,
                func_value,
            )

        raise ValueError(
            f"Unknown how_cmd params: {params} in how_cmd: {how_cmd}"
        )

    @staticmethod
    def get_colors_from_diversity(
        result_data, is_pew=False, pct_values_key="pct_values"
    ):

        if is_pew:
            return (
                ColorSpec.by_region_to_custom_value_with_custom_color_config(
                    Diversity.get_region_to_diversity(
                        result_data, is_pew, pct_values_key
                    ),
                    Diversity.RDI_BANDS,
                )
            )

        return ColorSpec.by_region_to_custom_value(
            Diversity.get_region_to_diversity(
                result_data, is_pew, pct_values_key
            ),
            False,
        )

    @staticmethod
    def get_color_spec_for_diversity_change(result_data, is_pew=False):
        region_to_diversity_change = Diversity.get_region_to_diversity_change(
            result_data, is_pew
        )
        return ColorSpec.by_region_to_custom_value(
            region_to_diversity_change, True
        )

    @staticmethod
    def get_color_spec_for_segregation(result_data):
        return ColorSpec.by_region_to_custom_value(
            Segregation.get_region_to_segregation(result_data), False
        )

    @staticmethod
    def get_color_spec_for_segregation_change(result_data):

        return ColorSpec.by_region_to_custom_value(
            Segregation.get_segregation_change(result_data), True
        )

    @staticmethod
    def get_region_to_change(dataset):
        region_to_change = {}
        for data in dataset.get_data_table():
            region_to_change[data["region_id"]] = data["change"]
        return region_to_change

    @staticmethod
    def get_color_spec_for_change(result_data):
        # change is always >= 0 (mean absolute pct-change), so use the
        # sequential (absolute) colour ramp instead of the diverging one.
        def value_mapper(change):
            return f"{change * 100:.2f}pp"

        return ColorSpec.by_region_to_custom_value(
            ColorSpecHelpers.get_region_to_change(result_data),
            False,
            value_mapper,
        )

    @staticmethod
    def get_colors_from_flips(dataset, idx=0):
        region_to_flip = {}
        for data in dataset.get_data_table():
            keys1 = list(data.get("values1", {}).keys())
            keys2 = list(data.get("values2", {}).keys())
            k1 = keys1[idx] if idx < len(keys1) else "(No Data)"
            k2 = keys2[idx] if idx < len(keys2) else "(No Data)"
            region_to_flip[data["region_id"]] = (
                f"{k1} to {k2}" if k1 != k2 else "(No Flip)"
            )
        return ColorSpec.by_region_to_custom_value(region_to_flip, True)
