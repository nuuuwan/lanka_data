from lanka_data.visual.plot.color_spec import ColorSpec
from lanka_data.visual.plot.color_spec.ColorSpecHelpers import ColorSpecHelpers
from utils_future import Log

log = Log("ColorSpecFactory")


class ColorSpecFactory:

    @staticmethod
    def get_color_spec(dataset, how_cmd) -> ColorSpec:
        is_diff = dataset.is_diff()
        if ":" in how_cmd:
            how_without_params, how_params = how_cmd.split(":")
        else:
            how_without_params = how_cmd
            how_params = None

        log.debug(f"{how_without_params=}, {how_params=}, {is_diff=}")

        has_values = dataset.has_values()
        if not has_values:
            return ColorSpec.by_custom_category_key(
                dataset,
                lambda data: data["region_id"],
                True,
            )

        if how_params == "Diversity":
            if is_diff:
                return ColorSpecHelpers.get_color_spec_for_diversity_change(
                    dataset,
                    is_pew=False,
                )

            return ColorSpecHelpers.get_colors_from_diversity(
                dataset,
                is_pew=False,
            )

        if how_params == "DiversityPew":
            if is_diff:
                return ColorSpecHelpers.get_color_spec_for_diversity_change(
                    dataset,
                    is_pew=True,
                )
            return ColorSpecHelpers.get_colors_from_diversity(
                dataset,
                is_pew=True,
            )

        if how_params == "Change":
            if is_diff:
                return ColorSpecHelpers.get_color_spec_for_change(dataset)
            return ColorSpecHelpers.get_color_spec_generic(
                dataset,
                how_without_params,
            )

        if how_params == "Segregation":
            if is_diff:
                return ColorSpecHelpers.get_color_spec_for_segregation_change(
                    dataset
                )
            return ColorSpecHelpers.get_color_spec_for_segregation(dataset)

        if is_diff:
            if (
                how_params in ColorSpecHelpers.KEY_PARAM_TO_I_RANK
                or how_params is None
            ):
                idx = ColorSpecHelpers.KEY_PARAM_TO_I_RANK.get(how_params, 0)
                return ColorSpecHelpers.get_colors_from_flips(dataset, idx=idx)

        return ColorSpecHelpers.get_color_spec_generic(dataset, how_cmd)
