from lanka_data.visual.annotations.AnnotationsStatMixin import (
    AnnotationsStatMixin,
)
from lanka_data.visual.annotations.NumberAbbreviator import NumberAbbreviator


class Annotations(AnnotationsStatMixin):
    def __init__(self, data_table):
        self.data_table = data_table or []

    @classmethod
    def from_data_table(cls, data_table):
        return cls(data_table)

    @staticmethod
    def _is_change_table(rows):
        return any(
            r["row"].get("values1") is not None
            or r["row"].get("change") is not None
            for r in rows
        )

    @staticmethod
    def _value_text(row_entry):
        display = row_entry["row"].get("display")
        if display:
            return display
        return NumberAbbreviator.format(row_entry["total"])

    @staticmethod
    def _extremes(rows):
        ordered = sorted(rows, key=lambda r: r["total"])
        low, high = ordered[0], ordered[-1]
        if low["name"] == high["name"]:
            return []
        return [
            f"Highest: {high['name']} " f"({Annotations._value_text(high)})",
            f"Lowest: {low['name']} " f"({Annotations._value_text(low)})",
        ]

    @staticmethod
    def _biggest_change(rows):
        top = max(rows, key=lambda r: abs(r["total"]))
        if top["total"] == 0:
            return []
        display = top["row"].get("display")
        text = display or NumberAbbreviator.signed(top["total"])
        return [f"Biggest change: {top['name']} " f"({text})"]

    def _outlier_callout(self, rows):
        outliers = self._outliers(rows)
        if not outliers:
            return []
        names = ", ".join(o["name"] for o in outliers)
        return [f"Outliers: {names}"]

    def callouts(self):
        rows = self._rows()
        if not rows:
            return []
        items = list(self._extremes(rows))
        if self._is_change_table(rows):
            items += self._biggest_change(rows)
        items += self._outlier_callout(rows)
        return items

    def summary(self):
        items = self.callouts()
        if not items:
            return ""
        return "What to notice — " + "; ".join(items)
