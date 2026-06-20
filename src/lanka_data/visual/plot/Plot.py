import os
import tempfile

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from lanka_data.visual.plot.Font import Font
from lanka_data.visual.plot.Footer import Footer
from lanka_data.visual.plot.Header import Header
from lanka_data.visual.plot.HeaderFooterBars import HeaderFooterBars
from lanka_data.visual.plot.SubFigureSpecs import SubFigureSpecs
from lanka_data.visual.plot.Text import Text
from utils_future import Log

log = Log("Plot")


class Plot:
    FIG_WIDTH = 16
    FIG_HEIGHT = 9
    FONT_FAMILY = "Fira Sans"
    DIR_OUTPUT = os.path.join(tempfile.gettempdir(), "lanka_data", "output")

    def __init__(self, visual):
        self.visual = visual

    def _draw_subfigures(self):
        figure_specs = SubFigureSpecs.get(self.command)
        n_figs = len(figure_specs)
        fig = plt.figure(figsize=(self.FIG_WIDTH, self.FIG_HEIGHT))

        outer_gs = gridspec.GridSpec(
            1, n_figs, figure=fig, top=0.92, bottom=0.08, wspace=0.25
        )

        result_data_list = []
        for j, (figure_label, command_for_subfigure) in enumerate(
            figure_specs.items()
        ):
            subfigure = fig.add_subfigure(outer_gs[0, j])
            renderer = self.renderer_class(
                figure_label,
                command_for_subfigure,
                self.is_cartogram,
                subfigure,
            )
            result_data = renderer.draw()
            result_data_list.append(result_data)

        return fig, result_data_list

    def draw(self):
        Font(self.FONT_FAMILY).install()

        fig = plt.gcf()

        HeaderFooterBars.draw_bars(fig)
        Header(self.visual).draw(
            lambda xy, text, fontsize, color, **kwargs: Text.plot(
                fig,
                xy,
                text,
                fontsize,
                color,
                **kwargs,
            )
        )
        Footer(self.visual).draw(
            lambda xy, text, fontsize, color, **kwargs: Text.plot(
                fig,
                xy,
                text,
                fontsize,
                color,
                **kwargs,
            )
        )

        image_dir = os.path.join(self.DIR_OUTPUT, self.visual.command.cmd_id)
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, "Image.png")
        fig.savefig(image_path, dpi=200, bbox_inches=0)
        plt.close(fig)

        log.debug(f"Wrote {image_path}")
        return {
            "image_path": image_path,
            "source_list": self.visual.get_source_list(),
        }
