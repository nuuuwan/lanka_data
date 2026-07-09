CATEGORY_MODIFIERS = {
    "1st": 1,
    "Top": 1,
    "Bottom": 1,
    "2nd": 2,
    "3rd": 3,
    "1stPct": 1,
    "2ndPct": 2,
    "3rdPct": 3,
    "Diversity": 1,
    "DiversityPew": 1,
}


class Modifiers:
    @classmethod
    def is_categorical_modifier(cls, modifier):
        return modifier in CATEGORY_MODIFIERS

    @classmethod
    def min_categories(cls, modifier):
        return CATEGORY_MODIFIERS.get(modifier, 0)

    @classmethod
    def is_change_modifier(cls, modifier):
        return modifier == "Change"
