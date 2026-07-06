from matplotlib.patches import Rectangle

from lanka_data.visual.plot.Style import Style


class TreeMapDrawMixin:
    EDGE_COLOR = "white"
    EDGE_WIDTH = 1.0
    MIN_LABEL_SIDE = 0.06

    @classmethod
    def _label_color(cls, color):
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        return "#333" if lum > 0.5 else "#eee"

    @classmethod
    def _draw_rect(cls, ax, rect, color, label, share):
        x, y, dx, dy = rect
        patch = Rectangle(
            (x, y),
            dx,
            dy,
            facecolor=color,
            edgecolor=cls.EDGE_COLOR,
            linewidth=cls.EDGE_WIDTH,
        )
        ax.add_patch(patch)
        if min(dx, dy) < cls.MIN_LABEL_SIDE:
            return
        fontsize = max(6, min(16, min(dx, dy) * 40))
        ax.text(
            x + dx / 2,
            y + dy / 2,
            f"{label}\n{share:.1%}",
            ha="center",
            va="center",
            fontsize=fontsize,
            color=cls._label_color(patch.get_facecolor()),
        )

    @classmethod
    def _draw_treemap(cls, ax, rects, labels, totals, category_to_color):
        grand_total = sum(totals.values()) or 1
        for rect, label in zip(rects, labels):
            share = totals[label] / grand_total
            color = category_to_color.get(label, "#999999")
            cls._draw_rect(ax, rect, color, label, share)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(
            "Composition Tree Map",
            fontsize=Style.FONT_SIZE_METADATA,
            color=Style.COLOR_METADATA,
        )
