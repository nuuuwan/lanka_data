from lanka_data.visual.plot.Label import Label
from lanka_data.visual.plot_visual.TriangleMapVisual.TriangleTextFit import (
    TriangleTextFit,
)


class TriangleMapLabelFitMixin:
    FIT_FONTSIZE = getattr(Label, "_fit_fontsize")

    @classmethod
    def _best_label_placement(cls, points, size, label, ax, fig):
        cx, cy, rect_w, rect_h, angle_deg = TriangleTextFit.best_label_fit(
            points, size
        )
        fontsize = cls.FIT_FONTSIZE(label, rect_w, rect_h, ax, fig)
        return cx, cy, angle_deg, fontsize
