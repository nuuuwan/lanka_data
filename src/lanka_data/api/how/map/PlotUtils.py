import os
import tempfile

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lanka_data.api.how.map.color_spec import ColorSpecFactory
from lanka_data.api.how.map.FontUtils import FontUtils
from lanka_data.api.how.map.GeoDataUtils import GeoDataUtils
from lanka_data.api.how.map.LabelUtils import LabelUtils
from lanka_data.api.how.map.LegendUtils import LegendUtils
from utils_future import Log

log = Log("PlotUtils")


class PlotUtils:
    DELIM_TITLE = " · "
    MAX_REGIONS_TO_LABEL = 30
    DEFAULT_EDGE_COLOR = "#fff"
    DEFAULT_EDGE_WIDTH = 0.2
    ASPECT_RATIO = 16 / 9
    FIG_WIDTH = 16
    FIG_HEIGHT = 9
    DIR_OUTPUT = os.path.join(
        tempfile.gettempdir(),
        "lanka_data",
        "output",
    )
    FONT_FAMILY = "Fira Sans"

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

    @staticmethod
    def get_figure_specs(command):
        from lanka_data.command.Command import Command

        when_cmd = command.when_cmd
        if "-" in when_cmd:
            when_parts = when_cmd.split("-")
            command1 = Command(
                command.what_cmd,
                when_parts[0],
                command.where_cmd,
                command.how_cmd,
            )
            command2 = Command(
                command.what_cmd,
                when_parts[1],
                command.where_cmd,
                command.how_cmd,
            )

            return {
                when_parts[0]: command1,
                when_parts[1]: command2,
                "Change": command,
            }

        return {"": command}

    # flake8: noqa: CFQ002
    @staticmethod
    def plot_subfigure(
        figure_label,
        command_for_subfigure,
        is_cartogram,
        subfig,
    ):
        how = command_for_subfigure.get_how()
        what = command_for_subfigure.get_what()
        when = command_for_subfigure.get_when()
        where = command_for_subfigure.get_where()

        result_data = how.get_data(what, when, where)
        data_list = result_data["data_list"]
        n_regions = len(data_list)
        gdf_region = GeoDataUtils.get_geopandas_dataframe(
            data_list, is_cartogram
        ).copy()
        region_color_map, value_to_color = ColorSpecFactory.get_color_spec(
            what, when, where, how
        ).unpack()
        gdf_region["color"] = gdf_region["region_id"].map(region_color_map)

        gs = subfig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
        ax = subfig.add_subplot(gs[0])
        legend_ax = subfig.add_subplot(gs[1])

        edge_color, edge_width = (
            PlotUtils.DEFAULT_EDGE_COLOR,
            PlotUtils.DEFAULT_EDGE_WIDTH,
        )

        gdf_region.plot(
            ax=ax,
            categorical=True,
            color=gdf_region["color"],
            edgecolor=edge_color,
            linewidth=edge_width,
        )
        if n_regions <= PlotUtils.MAX_REGIONS_TO_LABEL:
            LabelUtils.draw_labels(gdf_region, ax)
        LegendUtils.draw_legend(value_to_color, legend_ax)

        ax.set_axis_off()
        PlotUtils._plot_text(
            subfig,
            (0.5, 0.9),
            figure_label,
            fontsize=16,
            color="#000",
        )
        return result_data

    @staticmethod
    def plot_subfigures(command, is_cartogram):

        figure_specs = PlotUtils.get_figure_specs(command)

        n_figs = len(figure_specs)
        rows, cols = 1, n_figs
        fig = plt.figure(figsize=(PlotUtils.FIG_WIDTH, PlotUtils.FIG_HEIGHT))

        outer_gs = gridspec.GridSpec(rows, cols, figure=fig, top=1, bottom=0)
        subfigs_flat = [
            fig.add_subfigure(outer_gs[i, j])
            for i in range(rows)
            for j in range(cols)
        ]

        result_data_list = []
        for (figure_label, command_for_subfigure), subfig in zip(
            figure_specs.items(), subfigs_flat[:n_figs]
        ):

            result_data = PlotUtils.plot_subfigure(
                figure_label,
                command_for_subfigure,
                is_cartogram,
                subfig,
            )
            result_data_list.append(result_data)

        return fig, result_data_list

    @staticmethod
    def _draw_header(fig, command):
        HEADER_TITLE_DELIM = " · "
        header_title_items = [
            f"{command.get_what().title} ({command.get_when()})",
            command.get_where().get_description(),
            command.get_how().get_description(),
        ]
        header_title_items = [
            item.strip() for item in header_title_items if item.strip()
        ]

        PlotUtils._plot_text(
            fig,
            (0.5, 0.975),
            HEADER_TITLE_DELIM.join(header_title_items),
            16,
            "#fff",
        )

    @staticmethod
    def _draw_footer(fig, source_list):
        PlotUtils._plot_text(
            fig,
            (0.5, 0.025),
            "Data Sources: " + ", ".join(source_list),
            16,
            "#fff",
        )

    @staticmethod
    def _plot_rects(fig):
        rect = Rectangle(
            (0, 0),
            1,
            0.05,
            transform=fig.transFigure,
            facecolor="grey",
            edgecolor="none",
            zorder=0,
        )
        fig.patches.append(rect)
        rect = Rectangle(
            (0, 0.95),
            1,
            0.05,
            transform=fig.transFigure,
            facecolor="grey",
            edgecolor="none",
            zorder=0,
        )
        fig.patches.append(rect)

    @classmethod
    def draw_plot(cls, command, is_cartogram):
        FontUtils.install_font(cls.FONT_FAMILY)
        fig, result_data_list = PlotUtils.plot_subfigures(
            command,
            is_cartogram,
        )

        source_set = set()
        for result_data in result_data_list:
            source_set.add(result_data["source"])
        source_list = sorted(source_set)
        PlotUtils._plot_rects(fig)
        PlotUtils._draw_header(fig, command)
        PlotUtils._draw_footer(fig, source_list)

        image_dir = os.path.join(PlotUtils.DIR_OUTPUT, command.cmd_id)
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, "Image.png")

        fig.savefig(image_path, dpi=200, bbox_inches=0)
        plt.close(fig)

        log.debug(f"Wrote {image_path}")
        return {
            "image_path": image_path,
            "source_list": source_list,
        }
