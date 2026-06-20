from lanka_data.visual.plot.color_spec import ColorSpec
from lanka_data.visual.plot.color_spec.ColorSpecHelpers import ColorSpecHelpers


class ColorSpecFactory:

    @staticmethod
    def get_color_spec(dataset, how_cmd) -> ColorSpec:
        is_diff = dataset.is_diff()
        if ":" in how_cmd:
            how_without_params, how_params = how_cmd.split(":")

        else:
            how_without_params = how_cmd
            how_params = None

        values = (
            dataset.get_data_table()[0].get("values")
            if dataset.get_data_table()
            else None
        )

        if values is None:
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

        if how_params == "Flips":
            if is_diff:
                return ColorSpecHelpers.get_colors_from_flips(dataset)
            return ColorSpecHelpers.get_color_spec_generic(
                dataset,
                how_without_params,
            )

        if is_diff:
            idx = ColorSpecHelpers._PARAM_TO_IDX.get(how_params or "Top", 0)
            return ColorSpecHelpers.get_colors_from_flips(dataset, idx=idx)
        return ColorSpecHelpers.get_color_spec_generic(dataset, how_cmd)
