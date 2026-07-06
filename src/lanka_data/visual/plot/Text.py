class Text:
    @staticmethod
    def plot(
        fig, xy, text, fontsize, color, ha="center", va="center", **kwargs
    ):
        x, y = xy
        fig.text(
            x,
            y,
            text,
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            **kwargs,
        )
