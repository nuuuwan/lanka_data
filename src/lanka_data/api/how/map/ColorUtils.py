import matplotlib.pyplot as plt

from lanka_data.api.how.map.HueUtils import HueUtils


class ColorUtils:
    GROUP_TO_HUE_TO_LABEL_LIST = {
        "Religion": {
            HueUtils.GOLD: ["Buddhist"],
            HueUtils.ORANGE: ["Hindu"],
            HueUtils.GREEN: ["Islam"],
            HueUtils.BLUE: ["OtherChristian"],
            HueUtils.PURPLE: ["RomanCatholic"],
            HueUtils.GRAY: ["Other"],
        },
        "Ethnicity": {
            HueUtils.RED: ["Sinhalese"],
            HueUtils.ORANGE: ["SLTamil"],
            HueUtils.BLUE: ["IndMalaiyagaTamil"],
            HueUtils.GREEN: ["SLMoor"],
            HueUtils.TEAL: ["Malay"],
        },
        "Political Party": {
            HueUtils.PURPLE: ["SLPP", "SLMP"],
            HueUtils.BLUE: ["UPFA", "PA", "SLFP"],
            HueUtils.RED: ["NPP"],
            HueUtils.GREEN: ["SJB", "UNP", "NDF"],
            HueUtils.ORANGE: ["IND9", "ACTC", "ITAK"],
        },
    }

    HUE_IDX = {
        label: hue
        for group in GROUP_TO_HUE_TO_LABEL_LIST.values()
        for hue, labels in group.items()
        for label in labels
    }

    MIN_ALPHA = 0.33
    MAX_ALPHA = 1.0
    ALPHA_SPAN = MAX_ALPHA - MIN_ALPHA

    @staticmethod
    def _color_with_opacity(hex_color, pct):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        alpha = ColorUtils.MIN_ALPHA + pct * ColorUtils.ALPHA_SPAN
        return (r, g, b, alpha)

    @staticmethod
    def _is_light_color(color):
        if isinstance(color, str):
            color = color.lstrip("#")
            if len(color) == 6:
                r = int(color[0:2], 16) / 255
                g = int(color[2:4], 16) / 255
                b = int(color[4:6], 16) / 255
            else:
                return False
        else:
            if len(color) == 4 and color[3] < 0.4:
                return True
            r, g, b = color[0], color[1], color[2]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance > 0.45

    @staticmethod
    def rgb_to_hex(rgb):
        r, g, b = rgb

        def part(x):
            return f"{round(x * 255):02X}"

        return f"#{part(r)}{part(g)}{part(b)}"

    @staticmethod
    def p_to_color(p):
        return plt.cm.viridis(p)
