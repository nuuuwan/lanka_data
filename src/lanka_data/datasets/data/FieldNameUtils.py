from functools import cache


class FieldNameUtils:
    NORMALIZED_TO_ORIGINAL = {
        # Ethnicity
        "SLTamil": {"SriLankaTamil", "SLTamil"},
        "IndMalaiyagaTamil": {
            "IndianTamil",
            "IndTamil",
            "IndianTamilOrMalaiyagaThamilar",
            "IndianMalaiyagaTamil",
        },
        "SLMoor": {
            "SriLankaMoorOrMuslim",
            "SLMoor",
            "SriLankaMoorMuslim",
            "SriLankaMoor",
        },
        "SLChetty": {"SriLankaChetty", "SLChetty"},
        "Other": {"Other", "OtherEth"},
        # Cooking
        "Firewood": {"Firewood", "FireWood"},
        "Gas": {"Gas", "LpGas"},
        "SawdustPaddyHusk": {"SawdustPaddyHusk", "SawDust/PaddyHusk"},
        # Wall
        "Bricks": {"Bricks", "Brick"},
        # Roof
        "CadjanPalmyrah": {"CadjanPalmyrah", "Cadjan/Palmyrah"},
        "CadjanPalmyrahStraw": {
            "CadjanPalmyrahStraw",
            "Cadjan/Palmyrah/Straw",
        },
        # Energy
        "Kerosene": {"Kerosene", "KeroseneLamp"},
        "ElectricityGrid": {
            "ElectricityGrid",
            "ElectricityNationalElectricityNetwork",
        },
        # Marital Status
        "MarriedRegistered": {
            "MarriedRegistered",
            "Married((Registered)",
        },
        "MarriedCustomary": {"MarriedCustomary", "Married(Customary)"},
        "SeparatedNotLegally": {
            "SeparatedNotLegally",
            "SeperatedNotLegally",
            "Separated(NotLegally)",
        },
        "00-04 Years": {"0004"},
        "05-09 Years": {"0509"},
        "10-14 Years": {"1014"},
        "15-19 Years": {"1519"},
        "20-24 Years": {"2024"},
        "25-29 Years": {"2529"},
        "30-34 Years": {"3034"},
        "35-39 Years": {"3539"},
        "40-44 Years": {"4044"},
        "45-49 Years": {"4549"},
        "50-54 Years": {"5054"},
        "55-59 Years": {"5559"},
        "60-64 Years": {"6064"},
        "65-69 Years": {"6569"},
        "70-74 Years": {"7074"},
        "75-79 Years": {"7579"},
        "80-84 Years": {"8084"},
        "85+ Years": {"85+"},
    }
    NAME_IDX = {
        original_name.lower(): normalized_name
        for normalized_name, original_names in NORMALIZED_TO_ORIGINAL.items()
        for original_name in original_names
    }

    @staticmethod
    def from_snake_case_to_pascal_case(snake_str):
        return (
            snake_str.replace("_", " ")
            .title()
            .replace(" ", "")
            .replace("-", "")
        )

    @staticmethod
    @cache
    def normalize(field_name):
        remapped_name = FieldNameUtils.from_snake_case_to_pascal_case(
            field_name
        )
        return FieldNameUtils.NAME_IDX.get(
            remapped_name.lower(), remapped_name
        )
