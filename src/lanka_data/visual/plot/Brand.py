import matplotlib.pyplot as plt

from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot.Text import Text


class Brand:
    def __init__(self, visual):
        self.visual = visual

    def draw(self):
        fig = plt.gcf()
        y = (Style.FOOTER_HEIGHT + Style.BODY_TOP) / 2
        Text.plot(
            fig,
            (0.5, y),
            Style.BRAND_NAME.title(),
            fontsize=Style.FONT_SIZE_WATERMARK,
            color=Style.COLOR_WATERMARK,
            alpha=Style.WATERMARK_ALPHA,
            weight="bold",
            zorder=0,
        )
