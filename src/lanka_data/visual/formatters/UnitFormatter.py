class UnitFormatter:
    LABEL_TO_UNIT = {
        "RiverLen": "km",
        "Catchment": "sqkm",
    }

    def __init__(self, label):
        self.label = label

    def unit(self):
        return self.LABEL_TO_UNIT.get(self.label)

    def format(self, value_text):
        unit = self.unit()
        if not unit:
            return value_text
        return f"{value_text} {unit}"
