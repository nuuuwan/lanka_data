from lanka_data.visual.plot.color_spec import ColorSpec
from lanka_data.visual.plot.color_spec.ColorSpecHelpers import ColorSpecHelpers
from utils_future import Log

log = Log("ColorSpecFactory")


class ColorSpecFactory:

    @staticmethod
    def _get_diversity_spec(dataset, is_pew, is_diff):
        if is_diff:
            return ColorSpecHelpers.get_color_spec_for_diversity_change(
                dataset, is_pew=is_pew
            )
        return ColorSpecHelpers.get_colors_from_diversity(
            dataset, is_pew=is_pew
        )

    @staticmethod
    def _get_change_spec(dataset, how_without_params, is_diff):
        if is_diff:
            return ColorSpecHelpers.get_color_spec_for_change(dataset)
        return ColorSpecHelpers.get_color_spec_generic(
            dataset, how_without_params
        )

    @staticmethod
    def _get_segregation_spec(dataset, is_diff):
        if is_diff:
            return ColorSpecHelpers.get_color_spec_for_segregation_change(
                dataset
            )
        return ColorSpecHelpers.get_color_spec_for_segregation(dataset)

    @staticmethod
    def _get_change_or_segregation_spec(
        dataset, how_without_params, how_params, is_diff
    ):
        if how_params == "Change":
            return ColorSpecFactory._get_change_spec(
                dataset, how_without_params, is_diff
            )
        if how_params == "Segregation":
            return ColorSpecFactory._get_segregation_spec(dataset, is_diff)
        return None

    @staticmethod
    def _get_param_color_spec(
        dataset, how_without_params, how_params, is_diff
    ):
        if how_params == "Diversity":
            return ColorSpecFactory._get_diversity_spec(
                dataset, False, is_diff
            )
        if how_params == "DiversityPew":
            return ColorSpecFactory._get_diversity_spec(
                dataset, True, is_diff
            )
        return ColorSpecFactory._get_change_or_segregation_spec(
            dataset, how_without_params, how_params, is_diff
        )

    @staticmethod
    def _resolve_color_spec(
        dataset, how_without_params, how_params, is_diff, how_cmd
    ):
        spec = ColorSpecFactory._get_param_color_spec(
            dataset, how_without_params, how_params, is_diff
        )
        if spec is not None:
            return spec
        if is_diff and (
            how_params in ColorSpecHelpers.KEY_PARAM_TO_I_RANK
            or how_params is None
        ):
            idx = ColorSpecHelpers.KEY_PARAM_TO_I_RANK.get(how_params, 0)
            return ColorSpecHelpers.get_colors_from_flips(dataset, idx=idx)
        return ColorSpecHelpers.get_color_spec_generic(dataset, how_cmd)

    @staticmethod
    def get_color_spec(dataset, how_cmd) -> ColorSpec:
        is_diff = dataset.is_diff()
        if ":" in how_cmd:
            how_without_params, how_params = how_cmd.split(":")
        else:
            how_without_params = how_cmd
            how_params = None

        log.debug(f"{how_without_params=}, {how_params=}, {is_diff=}")

        if not dataset.has_values():
            return ColorSpec.by_custom_category_key(
                dataset,
                lambda data: data["region_id"],
                True,
            )
        return ColorSpecFactory._resolve_color_spec(
            dataset, how_without_params, how_params, is_diff, how_cmd
        )
