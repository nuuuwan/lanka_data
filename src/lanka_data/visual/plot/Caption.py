import matplotlib.pyplot as plt

from lanka_data.visual.annotations.Annotations import Annotations
from lanka_data.visual.plot.color_spec import ColorSpecFactory
from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot.Text import Text


class Caption:
    Y = Style.BODY_TOP + 0.02
    MAX_CHARS = 120

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

    def _summary(self):
        datasets = self.visual.datasets
        if not datasets:
            return ""
        rows = self._visible_rows(datasets[-1])
        return Annotations.from_data_table(rows).summary()

    def draw(self):
        text = self._summary()
        if not text:
            return
        if len(text) > self.MAX_CHARS:
            text = text[: self.MAX_CHARS - 1] + "…"
        Text.plot(
            plt.gcf(),
            (0.5, self.Y),
            text,
            fontsize=Style.FONT_SIZE_METADATA,
            color=Style.COLOR_METADATA,
        )
