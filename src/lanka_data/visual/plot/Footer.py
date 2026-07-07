import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lanka_data.visual.plot.Header import Header
from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot.Text import Text


class Footer:
    TITLE_DELIM = Header.TITLE_DELIM
    TEXT_COLOR = Style.COLOR_METADATA
    BACK_COLOR = Style.COLOR_SURFACE_FOOTER

    def __init__(self, visual):
        self.visual = visual

    def draw(self):
        fig = plt.gcf()

        fig.patches.append(
            Rectangle(
                (0, 0),
                1,
                Style.FOOTER_HEIGHT,
                transform=fig.transFigure,
                facecolor=self.BACK_COLOR,
                edgecolor=self.BACK_COLOR,
                zorder=0,
            )
        )

        footer_text = self.TITLE_DELIM.join(
            [
                "Original Data Sources: "
                + ", ".join(
                    [source.name for source in self.visual.get_sources()]
                ),
                "Structured Data & Code: " + Style.BRAND_URL,
            ]
        )

        Text.plot(
            fig,
            (0.5, Style.FOOTER_HEIGHT * 0.5),
            footer_text,
            fontsize=Style.FONT_SIZE_METADATA * 80 / len(footer_text),
            color=self.TEXT_COLOR,
        )
