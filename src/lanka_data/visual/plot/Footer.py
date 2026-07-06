import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot.Text import Text


class Footer:
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
                0.05,
                transform=fig.transFigure,
                facecolor=self.BACK_COLOR,
                edgecolor=self.BACK_COLOR,
                zorder=0,
            )
        )

        Text.plot(
            fig,
            (0.5, 0.025),
            "Data Sources: "
            + ", ".join(
                [source.name for source in self.visual.get_sources()]
            ),
            fontsize=Style.FONT_SIZE_METADATA,
            color=self.TEXT_COLOR,
        )
