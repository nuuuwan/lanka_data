from lanka_data.visual.plot.color_spec.ColorSpec.ColorSpec import (
    ColorSpec,
)

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
            return data["pct_values"][key]

        return f

    @staticmethod
    def get_color_spec_generic(dataset, how_cmd) -> ColorSpec:
        from lanka_data.datasets.command.fields.How import How

        how = How(how_cmd)
        params = how.modifier or "1st"
        if how.rank is not None:
            i_rank = how.rank
            func_key = ColorSpecHelpers.func_key_from_rank(i_rank)
            return ColorSpec.by_custom_category_key(
                dataset, func_key, hide_legend=False
            )
        if how.pct_rank is not None:
            key = how.pct_rank
            func_value = ColorSpecHelpers.func_pct_value_from_rank(key)
            return ColorSpec.by_single_pct_value(dataset, func_value)
        raise ValueError(
            f"Unknown how_cmd params: {params} in how_cmd: {how_cmd}"
        )
