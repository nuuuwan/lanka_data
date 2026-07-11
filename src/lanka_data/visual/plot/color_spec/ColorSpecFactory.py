from lanka_data.api.fields.How import How
from lanka_data.visual.plot.color_spec.ColorSpec.ColorSpec import ColorSpec
from lanka_data.visual.plot.color_spec.ColorSpecHelpers.ColorSpecHelpers import \
    ColorSpecHelpers
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
    def _get_param_color_spec(
        dataset, how_without_params, how_params, is_diff
    ):
        if how_params in ("Diversity", "DiversityPew"):
            return ColorSpecFactory._get_diversity_spec(
                dataset, how_params == "DiversityPew", is_diff
            )
        if how_params == "Change":
            return ColorSpecFactory._get_change_spec(
                dataset, how_without_params, is_diff
            )
        return None

    @staticmethod
    def _get_special_color_spec(dataset, how, is_diff):
        if how.is_top3:
            return ColorSpecHelpers.get_color_spec_for_top3(dataset)
        if how.is_cluster:
            return ColorSpecHelpers.get_color_spec_for_cluster(
                dataset, how.cluster_n
            )
        return ColorSpecFactory._get_param_color_spec(
            dataset, how.base, how.modifier, is_diff
        )

    @staticmethod
    def _resolve_color_spec(dataset, how, is_diff):
        spec = ColorSpecFactory._get_special_color_spec(dataset, how, is_diff)
        if spec is not None:
            return spec
        if is_diff and (how.rank is not None or how.modifier is None):
            idx = how.rank if how.rank is not None else 0
            return ColorSpecHelpers.get_colors_from_flips(dataset, idx=idx)
        return ColorSpecHelpers.get_color_spec_generic(dataset, how.value)

    @staticmethod
    def get_color_spec(dataset, how_cmd) -> ColorSpec:

        how = How(how_cmd)
        is_diff = dataset.is_diff()

        if not dataset.has_values():
            return ColorSpec.by_custom_category_key(
                dataset,
                lambda data: data["region_id"],
                True,
            )
        return ColorSpecFactory._resolve_color_spec(dataset, how, is_diff)
