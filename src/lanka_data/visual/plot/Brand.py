import matplotlib.pyplot as plt

from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot.Text import Text


class Brand:
    LETTER_SPACING = 4

    def __init__(self, visual):
        self.visual = visual

    def _spine_text(self):
        return (" " * self.LETTER_SPACING).join(Style.BRAND_NAME.upper())

    def draw(self):
        fig = plt.gcf()
        y = (Style.FOOTER_HEIGHT + Style.BODY_TOP) / 2
        Text.plot(
            fig,
            (Style.BRAND_SPINE_X, y),
            self._spine_text(),
            fontsize=Style.FONT_SIZE_BRAND,
            color=Style.COLOR_BRAND_SPINE,
            rotation=90,
            weight="bold",
        )
