from lanka_data.visual.plot.Label import Label
from lanka_data.visual.plot.LabelFit import LabelFit


class HexMapLabelFitMixin:
    FIT_FONTSIZE = getattr(Label, "_fit_fontsize")

    @staticmethod
    def _text_orientation(rect_w, rect_h, angle_deg):
        if rect_w >= rect_h:
            text_w, text_h, text_angle = rect_w, rect_h, angle_deg
        else:
            text_w, text_h, text_angle = rect_h, rect_w, angle_deg + 90.0
        while text_angle > 90.0:
            text_angle -= 180.0
        return text_w, text_h, text_angle

    @classmethod
    def _best_label_placement(cls, geom, label, ax, fig):
        cx, cy, rect_w, rect_h, angle_deg = LabelFit.best_label_fit(geom)
        text_w, text_h, text_angle = cls._text_orientation(
            rect_w, rect_h, angle_deg
        )
        fontsize = cls.FIT_FONTSIZE(label, text_w, text_h, ax, fig)
        return cx, cy, text_angle, fontsize
