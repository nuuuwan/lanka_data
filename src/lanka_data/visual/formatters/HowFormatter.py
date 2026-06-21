import re


class HowFormatter:
    BASE_LABELS = {
        "JSON": "JSON",
        "Map": "Basic Map",
        "Cartogram": "Cartogram (Population based)",
        "BarChart": "Bar Chart",
        "PieChart": "Pie Chart",
        "BumpChart": "Bump Chart",
    }

    PARAM_LABELS = {
        "2nd": "2nd most common",
        "3rd": "3rd most common",
        "Bottom": "Least common",
        "Top": "Most common",
    }

    def __init__(self, how_cmd):
        self.how_cmd = how_cmd

    @staticmethod
    def _split_camel(text):
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    def format(self) -> str:
        if ":" in self.how_cmd:
            base, param = self.how_cmd.split(":", 1)
            base_label = self.BASE_LABELS.get(base, self._split_camel(base))
            param_label = self.PARAM_LABELS.get(
                param, self._split_camel(param)
            )
            if base_label:
                return f"{base_label} by {param_label}"
            return param_label

        return self.BASE_LABELS.get(
            self.how_cmd, self._split_camel(self.how_cmd)
        )
