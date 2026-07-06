class NumberAbbreviator:
    @staticmethod
    def format(value):
        sign = "-" if value < 0 else ""
        av = abs(value)
        for divisor, suffix in ((1_000_000, "M"), (1_000, "K")):
            if av >= divisor:
                return f"{sign}{av / divisor:.1f}{suffix}"
        body = f"{av:.0f}" if av >= 10 else f"{av:.2f}"
        return f"{sign}{body}"

    @classmethod
    def signed(cls, value):
        prefix = "+" if value > 0 else ""
        return prefix + cls.format(value)
