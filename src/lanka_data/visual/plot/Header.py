import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lanka_data.visual.formatters.HowFormatter import HowFormatter
from lanka_data.visual.formatters.WhatFormatter import WhatFormatter
from lanka_data.visual.formatters.WhenFormatter import WhenFormatter
from lanka_data.visual.formatters.WhereFormatter import WhereFormatter
from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot.Text import Text


class Header:
    TITLE_DELIM = " · "
    TEXT_COLOR = Style.COLOR_TITLE
    BACK_COLOR = Style.COLOR_SURFACE_HEADER
    LINE_HEIGHT = 1

    def __init__(self, visual):
        self.visual = visual

    def _title(self):
        items = [
            WhatFormatter(self.visual.command.what_cmd).format(),
            WhenFormatter(self.visual.command.when_cmd).format(),
            WhereFormatter(self.visual.command.where_cmd).format(),
            HowFormatter(self.visual.command.how_cmd).format(),
        ]
        items = [item.strip() for item in items if item and item.strip()]
        return self.TITLE_DELIM.join(items) + self._correction_suffix()

    def _correction_suffix(self):
        note = getattr(self.visual.command, "correction_note", None)
        if not note:
            return ""
        return f"  ({note})"

    def _band_height(self, fig):
        line_pt = Style.FONT_SIZE_TITLE * self.LINE_HEIGHT
        line_frac = line_pt / (fig.get_figheight() * 72)
        return Style.MARGIN + line_frac

    def _draw_background(self, fig, band_bottom, band_h):
        fig.patches.append(
            Rectangle(
                (0, band_bottom),
                1,
                band_h,
                transform=fig.transFigure,
                facecolor=self.BACK_COLOR,
                edgecolor=self.BACK_COLOR,
                zorder=0,
            )
        )

    def draw(self):
        fig = plt.gcf()
        band_h = self._band_height(fig)
        band_bottom = 1 - band_h
        self._draw_background(fig, band_bottom, band_h)
        text = self._title()
        Text.plot(
            fig,
            (0.5, band_bottom + band_h / 2),
            text,
            fontsize=Style.FONT_SIZE_TITLE * 70 / len(text),
            color=self.TEXT_COLOR,
        )
