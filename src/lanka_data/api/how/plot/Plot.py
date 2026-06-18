import os
import tempfile

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from lanka_data.api.how.plot.Font import Font
from lanka_data.api.how.plot.Footer import Footer
from lanka_data.api.how.plot.Header import Header
from lanka_data.api.how.plot.HeaderFooterBars import HeaderFooterBars
from lanka_data.api.how.plot.SubFigureSpecs import SubFigureSpecs
from lanka_data.api.how.plot.Text import Text
from utils_future import Log

log = Log("Plot")


class Plot:
    FIG_WIDTH = 16
    FIG_HEIGHT = 9
    FONT_FAMILY = "Fira Sans"
    DIR_OUTPUT = os.path.join(tempfile.gettempdir(), "lanka_data", "output")

    def __init__(self, command, is_cartogram, renderer_class):
        self.command = command
        self.is_cartogram = is_cartogram
        self.renderer_class = renderer_class

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
        fig, result_data_list = self._draw_subfigures()

        source_set = set()
        for result_data in result_data_list:
            source_set.add(result_data["source"])
        source_list = sorted(source_set)

        HeaderFooterBars.draw_bars(fig)
        Header(self.command).draw(
            lambda xy, text, fontsize, color, **kwargs: Text.plot(
                fig,
                xy,
                text,
                fontsize,
                color,
                **kwargs,
            )
        )
        Footer(source_list).draw(
            lambda xy, text, fontsize, color, **kwargs: Text.plot(
                fig,
                xy,
                text,
                fontsize,
                color,
                **kwargs,
            )
        )

        image_dir = os.path.join(self.DIR_OUTPUT, self.command.cmd_id)
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, "Image.png")
        fig.savefig(image_path, dpi=200, bbox_inches=0)
        plt.close(fig)

        log.debug(f"Wrote {image_path}")
        return {
            "image_path": image_path,
            "source_list": source_list,
        }

    @classmethod
    def draw_plot(cls, command, is_cartogram, renderer_class):
        return cls(command, is_cartogram, renderer_class).draw()
