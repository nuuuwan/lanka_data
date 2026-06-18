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
        value_str = value_str.split("(")[0].strip()

        try:
            return float(value_str)
        except ValueError:
            return None
