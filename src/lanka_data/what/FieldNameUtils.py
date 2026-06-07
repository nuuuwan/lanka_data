from functools import cache


class FieldNameUtils:
    NORMALIZED_TO_ORIGINAL = {
        "SL Tamil": {"Sri Lanka Tamil", "SL Tamil"},
        "Ind/Malaiyaga Tamil": {
            "Ind Tamil",
            "Indian Tamil or Malaiyaga Thamilar",
        },
        "SL Moor": {"Sri Lanka Moor or Muslim", "SL Moor"},
    }
    NAME_IDX = {
        original_name: normalized_name
        for normalized_name, original_names in NORMALIZED_TO_ORIGINAL.items()
        for original_name in original_names
    }

    @staticmethod
    def from_snake_case_to_title_case(snake_str):
        return snake_str.replace("_", " ").title()

    @staticmethod
    @cache
    def normalize(field_name):
        remapped_name = FieldNameUtils.from_snake_case_to_title_case(
            field_name
        )
        return FieldNameUtils.NAME_IDX.get(remapped_name, remapped_name)
