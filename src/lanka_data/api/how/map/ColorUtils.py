import matplotlib.pyplot as plt


class ColorUtils:

    DEFAULT_CMAP_ABS = plt.cm.get_cmap("RdYlGn")
    DEFAULT_CMAP_DIFF = plt.cm.get_cmap("bwr")
    DEFAULT_CMAP_CAT = plt.cm.get_cmap("tab20")

    MIN_ALPHA = 0.33
    MAX_ALPHA = 1.0
    ALPHA_SPAN = MAX_ALPHA - MIN_ALPHA

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
    def p_to_color_for_abs(p):
        rgb = ColorUtils.DEFAULT_CMAP_ABS(p)
        hex = ColorUtils.rgb_to_hex(rgb[:3])
        return hex

    @staticmethod
    def p_to_color_for_diff(p):
        rgb = ColorUtils.DEFAULT_CMAP_DIFF(p)
        hex = ColorUtils.rgb_to_hex(rgb[:3])
        return hex

    @staticmethod
    def p_to_color_for_category(p):
        rgb = ColorUtils.DEFAULT_CMAP_CAT(p)
        hex = ColorUtils.rgb_to_hex(rgb[:3])
        return hex
