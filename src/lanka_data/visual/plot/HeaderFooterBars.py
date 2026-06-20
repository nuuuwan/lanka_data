from matplotlib.patches import Rectangle


class HeaderFooterBars:

    @staticmethod
    def draw_bars(fig):
        fig.patches.append(
            Rectangle(
                (0, 0),
                1,
                0.05,
                transform=fig.transFigure,
                facecolor="grey",
                edgecolor="none",
                zorder=0,
            )
        )
        fig.patches.append(
            Rectangle(
                (0, 0.95),
                1,
                0.05,
                transform=fig.transFigure,
                facecolor="grey",
                edgecolor="none",
                zorder=0,
            )
        )
