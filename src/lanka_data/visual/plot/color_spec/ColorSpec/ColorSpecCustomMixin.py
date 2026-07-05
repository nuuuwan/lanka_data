from utils_future import ColorUtils, Parse


class ColorSpecCustomMixin:
    @classmethod
    def _get_custom_value_and_color(
        cls, custom_value, p, has_non_float_values, is_diff, value_formatter
    ):
        if has_non_float_values:
            value = str(custom_value)
            color = cls.LABEL_TO_COLOR.get(custom_value) or (
                cls.p_to_color_for_category(p)
            )
            return value, color
        if is_diff:
            value = (
                value_formatter(custom_value)
                if value_formatter
                else f"{custom_value:+.4f}"
            )
            return value, cls.p_to_color_for_diff(p)
        value = (
            value_formatter(custom_value)
            if value_formatter
            else f"{custom_value:.4f}"
        )
        return value, cls.p_to_color_for_abs(p)

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
