import colorsys
import hashlib


def _hue_to_hex(hue: int) -> str:
    """Convert a hue (0–360°) at S=100%, L=50% to a hex colour string."""
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.5, 1.0)
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


class HUE:
    RED = 0
    DARK_RED = 345
    DARK_ORANGE = 20
    ORANGE = 35
    GOLD = 50
    GREEN = 120
    TEAL = 170
    BLUE = 240
    MAROON = 300
    GRAY = None  # achromatic – no hue


class ColorUtils:
    GROUP_TO_HUE_TO_LABEL_LIST = {
        "Religion": {
            HUE.GOLD: ["buddhist"],
            HUE.DARK_ORANGE: ["hindu"],
            HUE.TEAL: ["islam"],
            HUE.BLUE: ["other_christian"],
            HUE.MAROON: ["roman_catholic"],
            HUE.GRAY: ["other"],
        },
        "Ethnicity": {
            HUE.DARK_RED: ["sinhalese"],
            HUE.DARK_ORANGE: ["sl_tamil", "sri_lanka_tamil"],
            HUE.BLUE: ["ind_tamil", "indian_tamil_or_malaiyaga_thamilar"],
            HUE.TEAL: ["sl_moor", "sri_lanka_moor_or_muslim"],
            HUE.GREEN: ["malay"],
        },
        "Political Party": {
            HUE.DARK_RED: ["SLPP"],
            HUE.BLUE: ["UPFA", "PA", "SLFP"],
            HUE.RED: ["NPP"],
            HUE.GREEN: ["SJB", "UNP", "NDF"],
            HUE.ORANGE: ["IND9", "ACTC", "ITAK"],
            HUE.MAROON: ["SLMP"],
        },
    }

    HUE_IDX = {
        label: hue
        for group in GROUP_TO_HUE_TO_LABEL_LIST.values()
        for hue, labels in group.items()
        for label in labels
    }

    @staticmethod
    def get_random_color(label: str) -> str:
        digest = hashlib.md5(str(label).encode()).hexdigest()
        return f"#{digest[:6]}"

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
