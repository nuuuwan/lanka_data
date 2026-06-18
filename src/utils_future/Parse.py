class Parse:
    @staticmethod
    def float(value):
        value_str = (
            str(value)
            .replace(",", "")
            .replace("+", "")
            .replace("%", "")
            .replace("pp", "")
        )

        try:
            return float(value_str)
        except ValueError:
            return None
