import textwrap

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
    LINE_HEIGHT = 1.4
    CHAR_WIDTH_RATIO = 0.55

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
        return self.TITLE_DELIM.join(items)

    def _wrap(self, fig):
        width_pt = fig.get_figwidth() * 72 * (1 - 2 * Style.MARGIN)
        char_pt = Style.FONT_SIZE_TITLE * self.CHAR_WIDTH_RATIO
        max_chars = max(int(width_pt / char_pt), 8)
        return textwrap.wrap(self._title(), width=max_chars) or [""]

    def _line_frac(self, fig):
        line_pt = Style.FONT_SIZE_TITLE * self.LINE_HEIGHT
        return line_pt / (fig.get_figheight() * 72)

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

    def _draw_brand(self, fig):
        Text.plot(
            fig,
            (Style.MARGIN, 1 - Style.MARGIN / 2),
            Style.BRAND_NAME,
            fontsize=Style.FONT_SIZE_BRAND,
            color=Style.COLOR_BRAND,
            ha="left",
            weight="bold",
        )

    def _draw_title(self, fig, lines, band_bottom, line_frac):
        n_lines = len(lines)
        for i, line in enumerate(lines):
            y = band_bottom + (n_lines - i - 0.5) * line_frac
            Text.plot(
                fig,
                (0.5, y),
                line,
                fontsize=Style.FONT_SIZE_TITLE,
                color=self.TEXT_COLOR,
            )

    def draw(self):
        fig = plt.gcf()
        lines = self._wrap(fig)
        line_frac = self._line_frac(fig)
        band_h = Style.MARGIN + len(lines) * line_frac
        band_bottom = 1 - band_h
        self._draw_background(fig, band_bottom, band_h)
        self._draw_brand(fig)
        self._draw_title(fig, lines, band_bottom, line_frac)
