from lanka_data.api.data import Segregation
from lanka_data.datasets.data import Diversity
from lanka_data.visual.plot.color_spec.ColorSpec.ColorSpec import ColorSpec


class ColorSpecHelpersMixin:
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
        return ColorSpec.by_region_to_custom_value(
            Diversity.get_region_to_diversity_change(result_data, is_pew),
            True,
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
        return {
            data["region_id"]: data["change"]
            for data in dataset.get_data_table()
        }

    @staticmethod
    def get_color_spec_for_change(result_data):
        def value_mapper(change):
            return f"{change * 100:.2f}pp"

        return ColorSpec.by_region_to_custom_value(
            ColorSpecHelpersMixin.get_region_to_change(result_data),
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
