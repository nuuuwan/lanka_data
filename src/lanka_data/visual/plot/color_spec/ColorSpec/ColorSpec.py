from dataclasses import dataclass

from lanka_data.visual.plot.color_spec.ColorSpecConstants import (
    ColorSpecConstants,
)
from utils_future import ColorUtils, Parse

from .ColorSpecCategoryMixin import ColorSpecCategoryMixin
from .ColorSpecCustomMixin import ColorSpecCustomMixin


@dataclass
class ColorSpec(ColorSpecCategoryMixin, ColorSpecCustomMixin):
    region_to_color: dict[str, str]
    value_to_color: dict[str, str]
    value_to_region: dict[str, str] | None = None
    region_to_value: dict[str, float] | None = None
    region_to_value_str: dict[str, str] | None = None

    LABEL_TO_COLOR = {
        label: ColorUtils.hex_to_rgb(color)
        for label, color in ColorSpecConstants.LABEL_TO_COLOR.items()
    }

    @staticmethod
    def p_to_color_for_abs(p):
        return ColorSpecConstants.DEFAULT_CMAP_ABS(p)

    @classmethod
    def cmap_for_label(cls, label):
        rgb = cls.LABEL_TO_COLOR.get(label)
        if rgb is None:
            return ColorSpecConstants.DEFAULT_CMAP_ABS
        return ColorSpecConstants.build_cmap_for_color(rgb)

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
            if count == 0:
                continue
            expanded[f"{value} ({count})"] = color
        return expanded

    def unpack(self):
        is_float_values = (
            bool(self.value_to_color)
            and Parse.float(next(iter(self.value_to_color))) is not None
        )
        if is_float_values:
            return (
                self.region_to_color,
                self._unpack_float_values(self.value_to_color),
                self.value_to_region,
            )
        return (
            self.region_to_color,
            self._unpack_category_values(
                self.region_to_color, self.value_to_color
            ),
            None,
        )
