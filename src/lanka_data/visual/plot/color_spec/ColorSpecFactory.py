from lanka_data.api.what.DiffWhat import DiffWhat
from lanka_data.visual.plot.color_spec import ColorSpec
from lanka_data.visual.plot.color_spec.ColorSpecHelpers import ColorSpecHelpers


class ColorSpecFactory:

    @staticmethod
    def get_color_spec(command) -> ColorSpec:
        how = command.get_how()
        what = command.get_what()
        when = command.get_when()
        where = command.get_where()
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
                return ColorSpecHelpers.get_color_spec_for_diversity_change(
                    result_data,
                    is_pew=False,
                )

            return ColorSpecHelpers.get_colors_from_diversity(
                result_data,
                is_pew=False,
            )

        if how.params == "DiversityPew":
            if is_diff:
                return ColorSpecHelpers.get_color_spec_for_diversity_change(
                    result_data,
                    is_pew=True,
                )
            return ColorSpecHelpers.get_colors_from_diversity(
                result_data,
                is_pew=True,
            )

        if how.params == "Change":
            if is_diff:
                return ColorSpecHelpers.get_color_spec_for_change(result_data)
            return ColorSpecHelpers.get_color_spec_generic(
                result_data, how.without_params(), what
            )

        if how.params == "Segregation":
            if is_diff:
                return ColorSpecHelpers.get_color_spec_for_segregation_change(
                    result_data
                )
            return ColorSpecHelpers.get_color_spec_for_segregation(
                result_data
            )

        if how.params == "Flips":
            if is_diff:
                return ColorSpecHelpers.get_colors_from_flips(result_data)
            return ColorSpecHelpers.get_color_spec_generic(
                result_data, how.without_params(), what
            )

        if is_diff:
            idx = ColorSpecHelpers._PARAM_TO_IDX.get(how.params or "Top", 0)
            return ColorSpecHelpers.get_colors_from_flips(
                result_data, idx=idx
            )
        return ColorSpecHelpers.get_color_spec_generic(result_data, how, what)
