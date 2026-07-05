import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from datasets.visual.plot.Text import Text


class Footer:
    TEXT_COLOR = "#ccc"
    BACK_COLOR = "#fff"

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
            fontsize=16,
            color=self.TEXT_COLOR,
        )
