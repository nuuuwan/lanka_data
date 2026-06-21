import random
from dataclasses import dataclass

from lanka_data.visual.plot.color_spec.ColorSpecConstants import (
    ColorSpecConstants,
)
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

    def unpack(self):
        is_float_values = False
        if self.value_to_color:
            first_value = next(iter(self.value_to_color))
            is_float_values = Parse.float(first_value) is not None

        if is_float_values:

            sorted_value_to_color = dict(
                sorted(
                    self.value_to_color.items(),
                    key=lambda item: (
                        -(
                            Parse.float(item[0])
                            if Parse.float(item[0]) is not None
                            else float("-inf")
                        )
                    ),
                )
            )

            expanded_value_to_color = sorted_value_to_color
        else:
            color_to_count = {}
            for color in self.region_to_color.values():
                color_to_count[color] = color_to_count.get(color, 0) + 1

            sorted_value_to_color = None
            if self.value_to_color:
                sorted_value_to_color = dict(
                    sorted(
                        self.value_to_color.items(),
                        key=lambda item: (
                            (-color_to_count.get(item[1], 0), item[1])
                        ),
                    )
                )
            expanded_value_to_color = None
            if sorted_value_to_color:
                expanded_value_to_color = {}
                for value, color in sorted_value_to_color.items():
                    count = color_to_count.get(color, 0)
                    expanded_value = f"{value} ({count})"
                    expanded_value_to_color[expanded_value] = color

        return self.region_to_color, expanded_value_to_color

    @classmethod
    def by_custom_category_key(cls, dataset, func_key_getter, hide_legend):
        data_list = dataset.get_data_table()
        sorted_color_keys = sorted(
            list(set([func_key_getter(data) for data in data_list]))
        )
        random.shuffle(sorted_color_keys)

        n_keys = len(sorted_color_keys)
        region_to_color = {}
        value_to_color = {}
        for data in data_list:
            key = func_key_getter(data)
            if key in cls.LABEL_TO_COLOR:
                color = cls.LABEL_TO_COLOR[key]
            else:
                i_key = sorted_color_keys.index(key)
                p = i_key / (n_keys - 1) if n_keys > 1 else 0
                color = ColorSpec.p_to_color_for_category(p)

            region_id = data["region_id"]
            region_to_color[region_id] = color
            value_to_color[key] = color

        if len(data_list) >= 1 and "values" in data_list[0]:
            for k, v in data_list[0]["values"].items():
                if k not in value_to_color:
                    if k in cls.LABEL_TO_COLOR:
                        color = cls.LABEL_TO_COLOR[k]
                        value_to_color[k] = color

        if hide_legend:
            value_to_color = None
        return cls(region_to_color, value_to_color)

    @classmethod
    def by_single_pct_value(cls, dataset, how):
        data_list = dataset.get_data_table()
        is_diff = dataset.is_diff()
        single_pct_value = how.params

        pct_values = [
            data["pct_values"][single_pct_value] for data in data_list
        ]
        value_to_rank = {v: r for r, v in enumerate(sorted(set(pct_values)))}
        n = len(value_to_rank)
        value_to_color, region_to_color = {}, {}
        for data in data_list:
            value = data["pct_values"][single_pct_value]
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
    def by_region_to_custom_value(cls, region_to_custom_value, is_diff):
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
            [Parse.float(value) is None for value in sorted_custom_values]
        )

        n = len(sorted_custom_values)
        region_to_color = {}
        value_to_color = {}
        for region_id, custom_value in region_to_custom_value.items():
            i_values = sorted_custom_values.index(custom_value)
            p = i_values / (n - 1) if n > 1 else 0
            if has_non_float_values:
                value = str(custom_value)
                if custom_value in cls.LABEL_TO_COLOR:
                    color = cls.LABEL_TO_COLOR[custom_value]
                else:
                    color = ColorSpec.p_to_color_for_category(p)
            else:

                if is_diff:
                    value = f"{custom_value:+.4f}"
                    color = ColorSpec.p_to_color_for_diff(p)
                else:
                    value = f"{custom_value:.4f}"
                    color = ColorSpec.p_to_color_for_abs(p)

            region_to_color[region_id] = color
            value_to_color[value] = color
        return cls(region_to_color, value_to_color)

    @classmethod
    def by_region_to_custom_value_with_custom_color_config(
        cls, region_to_custom_value, custom_color_config
    ):
        region_to_color = {}
        value_to_color = {}
        for region_id, custom_value in region_to_custom_value.items():
            color = None
            value_label = None
            for (
                low,
                high,
                custom_color,
                custom_value_label,
            ) in custom_color_config:
                if low <= custom_value < high:
                    color = custom_color
                    value_label = custom_value_label
                    break
            if color is None:
                raise ValueError(
                    f"Custom value {custom_value} for region {region_id}"
                    + " does not fall into any specified range."
                )

            region_to_color[region_id] = color
            value_to_color[value_label] = color
        return cls(region_to_color, value_to_color)
