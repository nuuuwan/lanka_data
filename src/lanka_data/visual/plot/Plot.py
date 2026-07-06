import os
import tempfile

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from lanka_data.visual.plot.Font import Font
from lanka_data.visual.plot.Footer import Footer
from lanka_data.visual.plot.Header import Header
from lanka_data.visual.plot.Style import Style
from lanka_data.visual.plot.Text import Text
from utils_future import File, Log, timer

log = Log("Plot")


class Plot:
    FIG_WIDTH = 16
    FIG_HEIGHT = 9
    FONT_FAMILY = "Fira Sans"
    DIR_OUTPUT = os.path.join(tempfile.gettempdir(), "lanka_data", "output")

    def __init__(self, visual):
        self.visual = visual

    @timer
    def _draw_subfigures(self):
        n_figs = len(self.visual.datasets)
        fig = plt.figure(figsize=(self.FIG_WIDTH, self.FIG_HEIGHT))

        outer_gs = gridspec.GridSpec(
            1, n_figs, figure=fig, top=0.92, bottom=0.08, wspace=0.25
        )

        for i_dataset, dataset in enumerate(self.visual.datasets):
            sub_fig = fig.add_subfigure(outer_gs[0, i_dataset])
            self.visual.draw(dataset, sub_fig)
            Text.plot(
                sub_fig,
                (0.5, 0.9),
                dataset.get_year(),
                fontsize=Style.FONT_SIZE_PANEL,
                color=Style.COLOR_PANEL,
            )
        return fig

    def draw(self):
        Font(self.FONT_FAMILY).install()
        self._draw_subfigures()

        Header(self.visual).draw()
        Footer(self.visual).draw()

        image_dir = os.path.join(self.DIR_OUTPUT, self.visual.command.cmd_id)
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, "Image.png")
        plt.savefig(image_path, dpi=200, bbox_inches=0)
        plt.close("all")
        log.debug(f"Wrote {File(image_path)}")
        return {
            "image_path": image_path,
        }
