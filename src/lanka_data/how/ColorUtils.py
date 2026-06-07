import colorsys
import hashlib

DEFAULT_SATURATION = 1.0
DEFAULT_LIGHTNESS = 0.4
DEFAULT_GRAY = "#808080"
MIN_HUE_SEPARATION = 30


class HUE:
    RED = 0
    ORANGE = 30
    GOLD = 60
    GREEN = 120
    TEAL = 170
    BLUE = 240
    PURPLE = 300
    GRAY = None  # achromatic – no hue

    @staticmethod
    def _check_hue_separation() -> None:
        hues = sorted(
            v
            for k, v in vars(HUE).items()
            if not k.startswith("_") and isinstance(v, int)
        )
        for i, h1 in enumerate(hues):
            for h2 in hues[i + 1 :]:
                dist = min(h2 - h1, 360 - (h2 - h1))
                if dist < MIN_HUE_SEPARATION:
                    raise ValueError(
                        f"HUE values {h1}° and {h2}° are only {dist}° apart"
                        f" (minimum {MIN_HUE_SEPARATION}°)"
                    )

    @staticmethod
    def to_hex(hue) -> str:
        if hue is None:
            return DEFAULT_GRAY
        r, g, b = colorsys.hls_to_rgb(
            hue / 360.0, DEFAULT_LIGHTNESS, DEFAULT_SATURATION
        )
        return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


HUE._check_hue_separation()


class ColorUtils:
    GROUP_TO_HUE_TO_LABEL_LIST = {
        "Religion": {
            HUE.GOLD: ["Buddhist"],
            HUE.ORANGE: ["Hindu"],
            HUE.GREEN: ["Islam"],
            HUE.BLUE: ["OtherChristian"],
            HUE.PURPLE: ["RomanCatholic"],
            HUE.GRAY: ["Other"],
        },
        "Ethnicity": {
            HUE.RED: ["Sinhalese"],
            HUE.ORANGE: ["SLTamil"],
            HUE.BLUE: ["IndMalaiyagaTamil"],
            HUE.GREEN: ["SLMoor"],
            HUE.TEAL: ["Malay"],
        },
        "Political Party": {
            HUE.PURPLE: ["SLPP", "SLMP"],
            HUE.BLUE: ["UPFA", "PA", "SLFP"],
            HUE.RED: ["NPP"],
            HUE.GREEN: ["SJB", "UNP", "NDF"],
            HUE.ORANGE: ["IND9", "ACTC", "ITAK"],
        },
    }

    HUE_IDX = {
        label: hue
        for group in GROUP_TO_HUE_TO_LABEL_LIST.values()
        for hue, labels in group.items()
        for label in labels
    }

    @staticmethod
    def hue_to_hex(hue) -> str:
        return HUE.to_hex(hue)

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
