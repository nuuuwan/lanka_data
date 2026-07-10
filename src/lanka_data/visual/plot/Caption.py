import matplotlib.pyplot as plt

from lanka_data.visual.annotations.Annotations import Annotations
from lanka_data.visual.plot.color_spec.ColorSpecFactory import ColorSpecFactory
from lanka_data.visual.plot.InnerSquare import InnerSquare
from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot.Text import Text


class Caption:
    HEADING = "What to notice"
    X = InnerSquare.left + 0.01
    Y_TOP = InnerSquare.bottom + 0.22
    LINE_HEIGHT = 0.026
    HEADING_GAP = LINE_HEIGHT * 1.4
    LINE_SPACING = 1.6

    def __init__(self, visual):
        self.visual = visual

    @staticmethod
    def _region_names(dataset):
        names = {}
        for row in dataset.get_data_table():
            region_id = row.get("region_id")
            names[region_id] = row.get("region_name") or str(region_id)
        return names

    def _visible_rows(self, dataset):
        try:
            color_spec = ColorSpecFactory.get_color_spec(
                dataset, self.visual.how_cmd
            )
        except Exception:
            return []
        region_to_value = color_spec.region_to_value
        if not region_to_value:
            return []
        names = self._region_names(dataset)
        value_str = color_spec.region_to_value_str or {}
        return [
            {
                "region_name": names.get(region_id, str(region_id)),
                "total_value": value,
                "display": value_str.get(region_id),
            }
            for region_id, value in region_to_value.items()
        ]

    def _callouts(self):
        datasets = self.visual.datasets
        if not datasets:
            return []
        rows = self._visible_rows(datasets[-1])
        return Annotations.from_data_table(rows).callouts()

    def _summary(self):
        items = self._callouts()
        if not items:
            return ""
        return self.HEADING + " — " + "; ".join(items)

    def draw(self):
        items = self._callouts()
        if not items:
            return
        fig = plt.gcf()
        Text.plot(
            fig,
            (self.X, self.Y_TOP),
            self.HEADING,
            fontsize=Style.FONT_SIZE_METADATA,
            color=Style.COLOR_PANEL,
            ha="left",
            va="top",
            fontweight="bold",
        )
        Text.plot(
            fig,
            (self.X, self.Y_TOP - self.HEADING_GAP),
            "\n".join(items),
            fontsize=Style.FONT_SIZE_METADATA,
            color=Style.COLOR_METADATA,
            ha="left",
            va="top",
            linespacing=self.LINE_SPACING,
        )
