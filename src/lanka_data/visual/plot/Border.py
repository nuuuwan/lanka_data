import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lanka_data.visual.plot.InnerSquare import InnerSquare
from lanka_data.visual.plot.Style import Style


class Border:
    def __init__(self, visual):
        self.visual = visual

    def draw(self):
        fig = plt.gcf()
        pad = Style.BORDER_PAD
        fig.patches.append(
            Rectangle(
                (InnerSquare.left - pad, InnerSquare.bottom - pad),
                InnerSquare.width + 2 * pad,
                InnerSquare.height + 2 * pad,
                transform=fig.transFigure,
                facecolor="none",
                edgecolor=Style.COLOR_BORDER,
                linewidth=1,
                zorder=1,
            )
        )
