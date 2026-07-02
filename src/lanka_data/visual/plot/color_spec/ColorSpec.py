import random
from dataclasses import dataclass

from lanka_data.visual.plot.color_spec.ColorSpecConstants import \
    ColorSpecConstants
from utils_future import ColorUtils, Parse


@dataclass
class ColorSpec:
    region_to_color: dict[str, str]
    value_to_color: dict[str, str]

    LABEL_TO_COLOR = {
        label: ColorUtils.hex_to_rgb(color)
        for color, labels in ColorSpecConstants.COLOR_TO_LABELS.items()
        for label in labels
    }

    @staticmethod
    def p_to_color_for_abs(p):
        return ColorSpecConstants.DEFAULT_CMAP_ABS(p)

    @staticmethod
    def p_to_color_for_diff(p):
        return ColorSpecConstants.DEFAULT_CMAP_DIFF(p)

    @staticmethod
    def p_to_color_for_category(p):
        return ColorSpecConstants.DEFAULT_CMAP_CAT(p)

    @staticmethod
    def _unpack_float_values(value_to_color):
        return dict(
            sorted(
                value_to_color.items(),
                key=lambda item: -(
                    Parse.float(item[0])
                    if Parse.float(item[0]) is not None
                    else float("-inf")
                ),
            )
        )

    @staticmethod
    def _unpack_category_values(region_to_color, value_to_color):
        color_to_count = {}
        for color in region_to_color.values():
            color_to_count[color] = color_to_count.get(color, 0) + 1
        if not value_to_color:
            return None
        sorted_vtc = dict(
            sorted(
                value_to_color.items(),
                key=lambda item: ((-color_to_count.get(item[1], 0), item[1])),
            )
        )
        expanded = {}
        for value, color in sorted_vtc.items():
            count = color_to_count.get(color, 0)
            expanded[f"{value} ({count})"] = color
        return expanded

    def unpack(self):
        is_float_values = (
            bool(self.value_to_color)
            and Parse.float(next(iter(self.value_to_color))) is not None
        )
        if is_float_values:
            return self.region_to_color, self._unpack_float_values(
                self.value_to_color
            )
        return self.region_to_color, self._unpack_category_values(
            self.region_to_color, self.value_to_color
        )

    @classmethod
    def _get_category_color(cls, key, sorted_color_keys, n_keys):
        if key in cls.LABEL_TO_COLOR:
            return cls.LABEL_TO_COLOR[key]
        i_key = sorted_color_keys.index(key)
        p = i_key / (n_keys - 1) if n_keys > 1 else 0
        return ColorSpec.p_to_color_for_category(p)

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
                ColorSpec.p_to_color_for_abs(rank / (n - 1))
                if not is_diff
                else ColorSpec.p_to_color_for_diff(rank / (n - 1))
            )
            value_str = f"{value:+.1%}" if is_diff else f"{value:.1%}"
            if is_diff:
                value_str = value_str.replace("%", "pp")
            value_to_color[value_str] = color
            region_to_color[data["region_id"]] = color
        return cls(region_to_color, value_to_color)

    @classmethod
    def _get_custom_value_and_color(
        cls, custom_value, p, has_non_float_values, is_diff, value_formatter
    ):
        if has_non_float_values:
            value = str(custom_value)
            color = cls.LABEL_TO_COLOR.get(custom_value) or (
                ColorSpec.p_to_color_for_category(p)
            )
            return value, color
        if is_diff:
            value = (
                value_formatter(custom_value)
                if value_formatter
                else f"{custom_value:+.4f}"
            )
            return value, ColorSpec.p_to_color_for_diff(p)
        value = (
            value_formatter(custom_value)
            if value_formatter
            else f"{custom_value:.4f}"
        )
        return value, ColorSpec.p_to_color_for_abs(p)

    @classmethod
    def by_region_to_custom_value(
        cls, region_to_custom_value, is_diff, value_formatter=None
    ):
        sorted_custom_values = list(
            sorted(
                set(region_to_custom_value.values()),
                key=lambda v: (
                    Parse.float(v)
                    if Parse.float(v) is not None
                    else float("-inf")
                ),
            )
        )
        has_non_float_values = any(
            Parse.float(v) is None for v in sorted_custom_values
        )
        n = len(sorted_custom_values)
        region_to_color = {}
        value_to_color = {}
        for region_id, custom_value in region_to_custom_value.items():
            i_values = sorted_custom_values.index(custom_value)
            p = i_values / (n - 1) if n > 1 else 0
            value, color = cls._get_custom_value_and_color(
                custom_value,
                p,
                has_non_float_values,
                is_diff,
                value_formatter,
            )
            region_to_color[region_id] = color
            value_to_color[value] = color
        return cls(region_to_color, value_to_color)

    @classmethod
    def _find_custom_color(cls, custom_value, custom_color_config, region_id):
        if custom_value == "(No Data)":
            return (
                ColorUtils.rgb_to_hex(cls.LABEL_TO_COLOR[custom_value]),
                custom_value,
            )
        for (
            low,
            high,
            custom_color,
            custom_value_label,
        ) in custom_color_config:
            if low <= custom_value < high:
                return custom_color, custom_value_label
        raise ValueError(
            f"Custom value {custom_value} for region {region_id}"
            + " does not fall into any specified range."
        )

    @classmethod
    def by_region_to_custom_value_with_custom_color_config(
        cls, region_to_custom_value, custom_color_config
    ):
        region_to_color = {}
        value_to_color = {}
        for region_id, custom_value in region_to_custom_value.items():
            color, value_label = cls._find_custom_color(
                custom_value, custom_color_config, region_id
            )
            region_to_color[region_id] = color
            value_to_color[value_label] = color
        return cls(region_to_color, value_to_color)
