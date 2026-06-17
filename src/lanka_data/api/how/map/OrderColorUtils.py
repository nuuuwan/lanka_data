

class OrderColorUtils:
    _PARAM_TO_IDX = {"Top": 0, "2nd": 1, "3rd": 2, "Bottom": -1}

    @staticmethod
    def _func_key_getter(how, what):
        idx = OrderColorUtils._PARAM_TO_IDX.get(how.params or "Top")
        if idx is None:
            return None

        def func_key_getter(data):
            values = list(what.get_pct_values(data).keys())
            return values[idx] if idx < len(values) else '(No Data)'

        return func_key_getter
