from lanka_data.api.data import Diversity, Segregation
from lanka_data.api.how.map.color_spec import ColorSpec


class ColorSpecHelpers:
    _PARAM_TO_IDX = {"Top": 0, "2nd": 1, "3rd": 2, "Bottom": -1}

    @staticmethod
    def func_key_getter(how, what):
        idx = ColorSpecHelpers._PARAM_TO_IDX.get(how.params or "Top")
        if idx is None:
            return None

        def func_key_getter(data):
            values = list(what.get_pct_values(data).keys())
            return values[idx] if idx < len(values) else "(No Data)"

        return func_key_getter

    @staticmethod
    def get_color_spec_generic(result_data, how, what) -> ColorSpec:
        func_key_getter = ColorSpecHelpers.func_key_getter(how, what)
        if func_key_getter:
            return ColorSpec.by_custom_category_key(
                result_data, func_key_getter, False
            )
        return ColorSpec.by_single_pct_value(result_data, how)

    @staticmethod
    def get_colors_from_diversity(
        result_data, is_pew=False, pct_values_key="pct_values"
    ):

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
    def get_region_to_change(result_data):
        region_to_change = {}
        for data in result_data["data_list"]:
            region_to_change[data["region_id"]] = data["change"]
        return region_to_change

    @staticmethod
    def get_color_spec_for_change(result_data):
        return ColorSpec.by_region_to_custom_value(
            ColorSpecHelpers.get_region_to_change(result_data), True
        )

    @staticmethod
    def get_colors_from_flips(result_data):
        region_to_flip = {
            data["region_id"]: data["flip"]
            for data in result_data["data_list"]
        }
        return ColorSpec.by_region_to_custom_value(region_to_flip, True)
