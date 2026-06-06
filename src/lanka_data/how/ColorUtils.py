import random

random.seed(0)


class COLOR:
    GOLD = "#FFBE29"
    DARK_ORANGE = "#EB7400"
    TEAL = "#00534E"
    BLUE = "#0000FF"
    MAROON = "#800080"
    GRAY = "#808080"
    DARK_RED = "#8D153A"
    GREEN = "#008000"
    RED = "#FF0000"
    ORANGE = "#FFA500"


class ColorUtils:
    COLOR_IDX = {
        # Religion
        "buddhist": COLOR.GOLD,
        "hindu": COLOR.DARK_ORANGE,
        "islam": COLOR.TEAL,
        "other_christian": COLOR.BLUE,
        "roman_catholic": COLOR.MAROON,
        "other": COLOR.GRAY,
        # Ethnicity
        "sinhalese": COLOR.DARK_RED,
        "sl_tamil": COLOR.DARK_ORANGE,
        "sri_lanka_tamil": COLOR.DARK_ORANGE,
        "ind_tamil": COLOR.BLUE,
        "indian_tamil_or_malaiyaga_thamilar": COLOR.BLUE,
        "sl_moor": COLOR.TEAL,
        "sri_lanka_moor_or_muslim": COLOR.TEAL,
        "malay": COLOR.GREEN,
        # Political Party
        "SLPP": COLOR.DARK_RED,
        "UPFA": COLOR.BLUE,
        "PA": COLOR.BLUE,
        "SLFP": COLOR.BLUE,
        "NPP": COLOR.RED,
        "SJB": COLOR.GREEN,
        "UNP": COLOR.GREEN,
        "NDF": COLOR.GREEN,
        "IND9": COLOR.ORANGE,
        "SLMP": COLOR.MAROON,
        "ACTC": COLOR.ORANGE,
        "ITAK": COLOR.ORANGE,
    }

    @staticmethod
    def get_random_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    @staticmethod
    def _color_with_opacity(hex_color, pct):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        MIN_ALPHA = 0.25
        alpha = MIN_ALPHA + pct * (1.0 - MIN_ALPHA)
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
        return luminance > 0.5
