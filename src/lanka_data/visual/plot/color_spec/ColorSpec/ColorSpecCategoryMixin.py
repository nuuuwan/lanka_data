import random


class ColorSpecCategoryMixin:
    @classmethod
    def _get_category_color(cls, key, sorted_color_keys, n_keys):
        if key in cls.LABEL_TO_COLOR:
            return cls.LABEL_TO_COLOR[key]
        i_key = sorted_color_keys.index(key)
        p = i_key / (n_keys - 1) if n_keys > 1 else 0
        return cls.p_to_color_for_category(p)

    @classmethod
    def _fill_missing_value_colors(cls, data_list, value_to_color):
        if not data_list or "values" not in data_list[0]:
            return
        for k in data_list[0]["values"]:
            if k not in value_to_color and k in cls.LABEL_TO_COLOR:
                value_to_color[k] = cls.LABEL_TO_COLOR[k]

    @classmethod
    def by_custom_category_key(cls, dataset, func_key_getter, hide_legend):
        data_list = dataset.get_data_table()
        sorted_color_keys = sorted(
            set(func_key_getter(data) for data in data_list)
        )
        random.seed(0)
        random.shuffle(sorted_color_keys)
        n_keys = len(sorted_color_keys)
        region_to_color = {}
        value_to_color = {}
        for data in data_list:
            key = func_key_getter(data)
            color = cls._get_category_color(key, sorted_color_keys, n_keys)
            region_to_color[data["region_id"]] = color
            value_to_color[key] = color
        cls._fill_missing_value_colors(data_list, value_to_color)
        if hide_legend:
            value_to_color = None
        return cls(region_to_color, value_to_color)

    @classmethod
    def by_single_pct_value(cls, dataset, value_mapper):
        data_list = dataset.get_data_table()
        is_diff = dataset.is_diff()
        pct_values = [value_mapper(data) for data in data_list]
        value_to_rank = {v: r for r, v in enumerate(sorted(set(pct_values)))}
        n = len(value_to_rank)
        value_to_color, region_to_color = {}, {}
        for data in data_list:
            value = value_mapper(data)
            rank = value_to_rank[value]
            color = (
                cls.p_to_color_for_abs(rank / (n - 1))
                if not is_diff
                else cls.p_to_color_for_diff(rank / (n - 1))
            )
            value_str = f"{value:+.1%}" if is_diff else f"{value:.1%}"
            if is_diff:
                value_str = value_str.replace("%", "pp")
            value_to_color[value_str] = color
            region_to_color[data["region_id"]] = color
        return cls(region_to_color, value_to_color)
