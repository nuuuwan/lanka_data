class Text:
    @staticmethod
    def plot(fig, xy, text, fontsize, color, **kwargs):
        x, y = xy
        fig.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            **kwargs,
        )
