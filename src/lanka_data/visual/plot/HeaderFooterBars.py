from matplotlib.patches import Rectangle


class HeaderFooterBars:
    BACK_COLOR = "#eee"

    @staticmethod
    def draw_bars(fig):
        fig.patches.append(
            Rectangle(
                (0, 0),
                1,
                0.05,
                transform=fig.transFigure,
                facecolor=HeaderFooterBars.BACK_COLOR,
                edgecolor=HeaderFooterBars.BACK_COLOR,
                zorder=0,
            )
        )
        fig.patches.append(
            Rectangle(
                (0, 0.95),
                1,
                0.05,
                transform=fig.transFigure,
                facecolor=HeaderFooterBars.BACK_COLOR,
                edgecolor=HeaderFooterBars.BACK_COLOR,
                zorder=0,
            )
        )
