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

    def __init__(self, visual):
        self.visual = visual

    def draw(self):
        fig = plt.gcf()

        fig.patches.append(
            Rectangle(
                (0, 0.95),
                1,
                0.05,
                transform=fig.transFigure,
                facecolor=self.BACK_COLOR,
                edgecolor=self.BACK_COLOR,
                zorder=0,
            )
        )

        header_title_items = [
            WhatFormatter(self.visual.command.what_cmd).format(),
            WhenFormatter(self.visual.command.when_cmd).format(),
            WhereFormatter(self.visual.command.where_cmd).format(),
            HowFormatter(self.visual.command.how_cmd).format(),
        ]
        header_title_items = [
            item.strip()
            for item in header_title_items
            if item and item.strip()
        ]
        Text.plot(
            fig,
            (0.5, 0.975),
            self.TITLE_DELIM.join(header_title_items),
            fontsize=Style.FONT_SIZE_TITLE,
            color=self.TEXT_COLOR,
        )
