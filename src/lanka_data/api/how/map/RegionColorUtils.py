from lanka_data.api.how.map import ColorSpec
from lanka_data.api.how.map.Diversity import Diversity
from lanka_data.api.how.map.OrderColorUtils import OrderColorUtils
from lanka_data.api.how.map.Segregation import Segregation
from lanka_data.api.what.DiffWhat import DiffWhat


class RegionColorUtils:

    @staticmethod
    def get_color_spec_generic(result_data, how, what) -> ColorSpec:
        func_key_getter = OrderColorUtils._func_key_getter(how, what)
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
            RegionColorUtils.get_region_to_change(result_data), True
        )

    @staticmethod
    def get_colors_from_flips(result_data):
        region_to_flip = {
            data["region_id"]: data["flip"]
            for data in result_data["data_list"]
        }
        return ColorSpec.by_region_to_custom_value(region_to_flip, True)

    @staticmethod
    def get_color_spec(what, when, where, how) -> ColorSpec:
        result_data = how.get_data(what, when, where)
        data_list = result_data["data_list"]
        is_diff = isinstance(what, DiffWhat)

        if what.get_values(data_list[0]) is None:
            return ColorSpec.by_custom_category_key(
                result_data,
                lambda data: data["region_id"],
                True,
            )

        if how.params == "Diversity":
            if is_diff:
                return RegionColorUtils.get_color_spec_for_diversity_change(
                    result_data,
                    is_pew=False,
                )

            return RegionColorUtils.get_colors_from_diversity(
                result_data,
                is_pew=False,
            )

        if how.params == "DiversityPew":
            if is_diff:
                return RegionColorUtils.get_color_spec_for_diversity_change(
                    result_data,
                    is_pew=True,
                )
            return RegionColorUtils.get_colors_from_diversity(
                result_data,
                is_pew=True,
            )

        if how.params == "Change":
            if is_diff:
                return RegionColorUtils._colors_with_change(result_data)
            return RegionColorUtils.get_color(
                result_data, how.without_params(), what
            )

        if how.params == "Segregation":
            if is_diff:
                return RegionColorUtils.get_color_spec_for_segregation_change(
                    result_data
                )
            return RegionColorUtils.get_color_spec_for_segregation(
                result_data
            )

        if how.params == "Flips":
            if is_diff:
                return RegionColorUtils.get_colors_from_flips(result_data)
            return RegionColorUtils.get_color_spec_generic(
                result_data, how.without_params(), what
            )

        return RegionColorUtils.get_color_spec_generic(result_data, how, what)
