import matplotlib.pyplot as plt

from lanka_data.visual.annotations.Annotations import Annotations
from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot.Text import Text


class Caption:
    Y = Style.BODY_TOP + 0.02
    MAX_CHARS = 120

    def __init__(self, visual):
        self.visual = visual

    def _summary(self):
        datasets = self.visual.datasets
        if not datasets:
            return ""
        try:
            data_table = datasets[-1].get_data_table()
        except Exception:
            return ""
        return Annotations.from_data_table(data_table).summary()

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
