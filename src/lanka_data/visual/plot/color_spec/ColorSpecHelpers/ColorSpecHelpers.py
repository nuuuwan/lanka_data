from lanka_data.visual.plot.color_spec.ColorSpec.ColorSpec import ColorSpec

from .ColorSpecHelpersMixin import ColorSpecHelpersMixin


class ColorSpecHelpers(ColorSpecHelpersMixin):
    KEY_PARAM_TO_I_RANK = {
        "1st": 0,
        "Top": 0,
        "2nd": 1,
        "3rd": 2,
        "Bottom": -1,
    }
    PCT_VALUE_PARAM_TO_KEY = {"1stPct": 0, "2ndPct": 1, "3rdPct": 2}

    @staticmethod
    def func_key_from_rank(i_rank):
        def f(data):
            keys = list(data["pct_values"].keys())
            return keys[i_rank] if i_rank < len(keys) else "(No Data)"

        return f

    @staticmethod
    def func_pct_value_from_rank(i_rank):
        def f(data):
            return list(data["pct_values"].values())[i_rank]

        return f

    @staticmethod
    def func_pct_value_from_key(key):
        def f(data):
            return data["pct_values"].get(key, 0)

        return f

    @staticmethod
    def _available_category_keys(dataset):
        get_category_keys = getattr(dataset, "get_category_keys", None)
        if get_category_keys is not None:
            return get_category_keys()
        keys = set()
        for data in dataset.get_data_table():
            keys.update(data.get("pct_values", {}).keys())
        return keys

    @staticmethod
    def _get_category_spec(dataset, category) -> ColorSpec:
        if category not in ColorSpecHelpers._available_category_keys(dataset):
            raise ValueError(f"Unknown category: {category}")
        func_value = ColorSpecHelpers.func_pct_value_from_key(category)
        return ColorSpec.by_single_pct_value(
            dataset, func_value, label=category
        )

    @staticmethod
    def get_color_spec_generic(dataset, how_cmd) -> ColorSpec:
        from lanka_data.api.fields.How import How

        how = How(how_cmd)
        if how.pct_rank is not None:
            func_value = ColorSpecHelpers.func_pct_value_from_rank(
                how.pct_rank
            )
            return ColorSpec.by_single_pct_value(dataset, func_value)
        if how.category is not None:
            return ColorSpecHelpers._get_category_spec(dataset, how.category)
        i_rank = how.rank if how.rank is not None else 0
        func_key = ColorSpecHelpers.func_key_from_rank(i_rank)
        return ColorSpec.by_custom_category_key(
            dataset, func_key, hide_legend=False
        )
