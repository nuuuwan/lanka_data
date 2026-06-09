from functools import cache


class FieldNameUtils:
    NORMALIZED_TO_ORIGINAL = {
        "SLTamil": {"SriLankaTamil", "SLTamil"},
        "IndTamil": {
            "IndTamil",
            "IndianTamilOrMalaiyagaThamilar",
        },
        "SLMoor": {"SriLankaMoorOrMuslim", "SLMoor"},
        "SLChetty": {"SriLankaChetty", "SLChetty"},
    }
    NAME_IDX = {
        original_name.lower(): normalized_name
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
        return FieldNameUtils.NAME_IDX.get(
            remapped_name.lower(), remapped_name
        )
