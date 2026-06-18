from dataclasses import dataclass

import matplotlib.pyplot as plt

from utils_future import Parse


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i: i + 2], 16) / 256.0 for i in (0, 2, 4))


@dataclass
class ColorSpec:
    region_to_color: dict[str, str]
    value_to_color: dict[str, str]

    DEFAULT_CMAP_ABS = plt.cm.get_cmap("YlGn")
    DEFAULT_CMAP_DIFF = plt.cm.get_cmap("coolwarm")
    DEFAULT_CMAP_CAT = plt.cm.get_cmap("rainbow")

    COLOR_TO_LABELS = {
        # Religion & Ethnicity
        "#FFBE29": ["Buddhist"],
        "#EB7400": ["Hindu", "SLTamil"],
        "#00534E": ["Islam", "SLMuslim"],
        "#8D153A": ["Sinhalese"],
        "#2000c0": ["OtherChristian"],
        "#c000c0": ["RomanCatholic"],
        # Null
        "#eeeeee": ["(No Data)"],
        "#dddddd": ["Other"],
        "#cccccc": ["(No Flip)"],
        # Political Parties
        "#222288": ["SLFP", "PA", "UPFA"],
        "#004400": ["ACMC", "MNA", "NC", "SLMC", "NUA"],
        "#008800": ["UNP", "NDF", "SJB"],
        "#009900": [],
        "#880000": ["SLPP", "OPPP"],
        "#880088": ["SLMP"],
        "#e0e0e0": ["IG", "IG2", "IG3"],
        "#8800ff": ["DUNF"],
        "#0088ff": ["SB"],
        "#ff0000": [
            "JVP",
            "NMPP",
            "NPP",
            "MEP",
            "USA",
            "SLPF",
            "DNA",
            "JJB",
            "LSSP",
            "CP",
            "NSSP",
        ],
        "#ff2200": [
            "ELMSP",
            "EPDP",
            "TMVP",
            "EROS",
        ],
        "#ff4400": ["CWC", "UPF"],
        "#ffcc00": ["SU", "JHU"],
        "#ffdd00": ["AITC", "ITAK", "TULF", "ACTC", "IND9"],
        "#ffffff": ["ELJP", "INDI"],
        "#ff8822": ["IND16"],
        # Validation only
        "#0088f1": ["A"],
        "#ff4401": ["B"],
    }

    LABEL_TO_COLOR = {
        label: hex_to_rgb(color)
        for color, labels in COLOR_TO_LABELS.items()
        for label in labels
    }

    @staticmethod
    def p_to_color_for_abs(p):
        return ColorSpec.DEFAULT_CMAP_ABS(p)

    @staticmethod
    def p_to_color_for_diff(p):
        return ColorSpec.DEFAULT_CMAP_DIFF(p)

    @staticmethod
    def p_to_color_for_category(p):
        return ColorSpec.DEFAULT_CMAP_CAT(p)

    def unpack(self):
        color_to_count = {}
        for region, color in self.region_to_color.items():
            color_to_count[color] = color_to_count.get(color, 0) + 1

        sorted_value_to_color = dict(
            sorted(
                self.value_to_color.items(),
                key=lambda item: (
                    -color_to_count[item[1]],
                    -(
                        Parse.float_float(item[0])
                        if Parse.float_float(item[0]) is not None
                        else 0
                    ),
                ),
            )
        )

        expanded_value_to_color = {}
        for value, color in sorted_value_to_color.items():
            count = color_to_count.get(color, 0)
            expanded_value = f"{value} ({count})"
            expanded_value_to_color[expanded_value] = color

        return self.region_to_color, expanded_value_to_color

    @classmethod
    def by_custom_category_key(
        cls, result_data, func_key_getter, hide_legend
    ):
        data_list = result_data["data_list"]
        sorted_color_keys = sorted(
            list(set([func_key_getter(data) for data in data_list]))
        )

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

        if hide_legend:
            value_to_color = None
        return cls(region_to_color, value_to_color)

    @classmethod
    def by_single_pct_value(cls, result_data, how):
        is_diff = "flip" in result_data["data_list"][0]
        single_pct_value = how.params
        data_list = result_data["data_list"]
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
            sorted(set(region_to_custom_value.values()))
        )
        has_non_float_values = any(
            [
                Parse.float_float(value) is None
                for value in sorted_custom_values
            ]
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
