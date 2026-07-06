from lanka_data.visual.plot.Label import Label
from lanka_data.visual.plot_visual.HexMapVisual.HexTextFit import HexTextFit


class HexMapLabelFitMixin:
    FIT_FONTSIZE = getattr(Label, "_fit_fontsize")

    @classmethod
    def _best_label_placement(cls, points, radius, label, ax, fig):
        cx, cy, rect_w, rect_h, angle_deg = HexTextFit.best_label_fit(
            points, radius
        )
        fontsize = cls.FIT_FONTSIZE(label, rect_w, rect_h, ax, fig)
        return cx, cy, angle_deg, fontsize
