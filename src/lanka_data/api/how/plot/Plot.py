import os
import tempfile

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lanka_data.api.how.plot.Font import Font
from lanka_data.api.how.plot.Footer import Footer
from lanka_data.api.how.plot.Header import Header
from lanka_data.api.how.plot.SubFigure import SubFigure
from utils_future import Log

log = Log("Plot")


class HeaderFooterBars:

    @staticmethod
    def _draw_bars(fig):
        fig.patches.append(
            Rectangle(
                (0, 0),
                1,
                0.05,
                transform=fig.transFigure,
                facecolor="grey",
                edgecolor="none",
                zorder=0,
            )
        )
        fig.patches.append(
            Rectangle(
                (0, 0.95),
                1,
                0.05,
                transform=fig.transFigure,
                facecolor="grey",
                edgecolor="none",
                zorder=0,
            )
        )


class Plot:
    FIG_WIDTH = 16
    FIG_HEIGHT = 9
    FONT_FAMILY = "Fira Sans"
    DIR_OUTPUT = os.path.join(
        tempfile.gettempdir(),
        "lanka_data",
        "output",
    )

    def __init__(self, command, is_cartogram):
        self.command = command
        self.is_cartogram = is_cartogram

    @staticmethod
    def _plot_text(fig, xy, text, fontsize, color, **kwargs):
        x, y = xy
        fig.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            **kwargs,
        )

    def _draw_subfigures(self):
        figure_specs = SubFigure._get_figure_specs(self.command)
        n_figs = len(figure_specs)
        fig = plt.figure(figsize=(self.FIG_WIDTH, self.FIG_HEIGHT))

        outer_gs = gridspec.GridSpec(1, n_figs, figure=fig, top=1, bottom=0)
        subfigs_flat = [
            fig.add_subfigure(outer_gs[0, j]) for j in range(n_figs)
        ]

        result_data_list = []
        for (figure_label, command_for_subfigure), subfigure in zip(
            figure_specs.items(), subfigs_flat[:n_figs]
        ):
            sub_figure = SubFigure(
                figure_label,
                command_for_subfigure,
                self.is_cartogram,
                subfigure,
            )

            def subfigure_text(
                xy,
                text,
                fontsize,
                color,
                sf=subfigure,
                **kwargs,
            ):
                return self._plot_text(
                    sf,
                    xy,
                    text,
                    fontsize,
                    color,
                    **kwargs,
                )

            result_data = sub_figure.draw(subfigure_text)
            result_data_list.append(result_data)

        return fig, result_data_list

    def draw(self):
        Font(self.FONT_FAMILY).install()
        fig, result_data_list = self._draw_subfigures()

        source_set = set()
        for result_data in result_data_list:
            source_set.add(result_data["source"])
        source_list = sorted(source_set)

        HeaderFooterBars._draw_bars(fig)
        Header(self.command).draw(
            lambda xy, text, fontsize, color, **kwargs: self._plot_text(
                fig,
                xy,
                text,
                fontsize,
                color,
                **kwargs,
            )
        )
        Footer(source_list).draw(
            lambda xy, text, fontsize, color, **kwargs: self._plot_text(
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
    def draw_plot(cls, command, is_cartogram):
        return cls(command, is_cartogram).draw()
