import colorsys


class HueUtils:
    # hues
    RED = 0
    ORANGE = 30
    GOLD = 60
    GREEN = 120
    TEAL = 170
    BLUE = 240
    PURPLE = 300
    GRAY = None  # achromatic – no hue

    # params
    DEFAULT_SATURATION = 1.0
    DEFAULT_LIGHTNESS = 0.4
    DEFAULT_GRAY = "#808080"

    @staticmethod
    def to_hex(hue) -> str:
        if hue is None:
            return HueUtils.DEFAULT_GRAY
        r, g, b = colorsys.hls_to_rgb(
            hue / 360.0,
            HueUtils.DEFAULT_LIGHTNESS,
            HueUtils.DEFAULT_SATURATION,
        )
        return (
            f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"
        )
