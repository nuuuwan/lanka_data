import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from lanka_data.visual.plot.Brand import Brand
from lanka_data.visual.plot.Caption import Caption
from lanka_data.visual.plot.Font import Font
from lanka_data.visual.plot.Footer import Footer
from lanka_data.visual.plot.Header import Header
from lanka_data.visual.plot.InnerSquare import InnerSquare
from lanka_data.visual.plot.PlotLayout import PlotLayout
from lanka_data.visual.plot.QRCode import QRCode
from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot.Text import Text
from utils_future import File, Log, timer

log = Log("Plot")


class Plot:
    FONT_FAMILY = "Fira Sans"
    DIR_OUTPUT = "_output"

    def __init__(self, visual):
        self.visual = visual

    @timer
    def _draw_subfigures(self):
        layout = PlotLayout(len(self.visual.datasets))
        fig = plt.figure(figsize=layout.figsize)

        outer_gs = gridspec.GridSpec(
            layout.n_rows,
            layout.n_cols,
            figure=fig,
            left=InnerSquare.left,
            right=InnerSquare.right,
            top=InnerSquare.top,
            bottom=InnerSquare.bottom,
            wspace=0.25,
        )

        for i_dataset, dataset in enumerate(self.visual.datasets):
            row, col = layout.position(i_dataset)
            sub_fig = fig.add_subfigure(outer_gs[row, col])
            self.visual.draw(dataset, sub_fig)
            panel_label = (
                getattr(dataset, "panel_label", None) or dataset.get_year()
            )
            Text.plot(
                sub_fig,
                (0.5, 0.9),
                panel_label,
                fontsize=Style.FONT_SIZE_PANEL,
                color=Style.COLOR_PANEL,
            )
        return fig

    def draw(self):
        Font(self.FONT_FAMILY).install()
        self._draw_subfigures()

        Header(self.visual).draw()
        QRCode(self.visual).draw()
        Footer(self.visual).draw()
        Caption(self.visual).draw()
        Brand(self.visual).draw()

        image_dir = os.path.join(self.DIR_OUTPUT, self.visual.command.cmd_id)
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, "Image.png")
        plt.savefig(image_path, dpi=200, bbox_inches=0)
        svg_path = os.path.join(image_dir, "Image.svg")
        plt.savefig(svg_path, bbox_inches=0)
        plt.close("all")
        log.debug(f"Wrote {File(image_path)}")
        log.debug(f"Wrote {File(svg_path)}")
        return {
            "image_path": image_path,
            "svg_path": svg_path,
        }
