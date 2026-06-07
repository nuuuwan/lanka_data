from functools import cache


class FieldNameUtils:
    NORMALIZED_TO_ORIGINAL = {
        "SLTamil": {"Sri Lanka Tamil", "SL Tamil"},
        "IndMalaiyagaTamil": {
            "Ind Tamil",
            "Indian Tamil Or Malaiyaga Thamilar",
        },
        "SLMoor": {"Sri Lanka Moor Or Muslim", "SL Moor"},
        "SLChetty": {"Sri Lanka Chetty"},
    }
    NAME_IDX = {
        original_name: normalized_name
        for normalized_name, original_names in NORMALIZED_TO_ORIGINAL.items()
        for original_name in original_names
    }

    @staticmethod
    def from_snake_case_to_pascal_case(snake_str):
        return snake_str.replace("_", " ").title().replace(" ", "")

    @staticmethod
    @cache
    def normalize(field_name):
        remapped_name = FieldNameUtils.from_snake_case_to_pascal_case(
            field_name
        )
        return FieldNameUtils.NAME_IDX.get(remapped_name, remapped_name)
