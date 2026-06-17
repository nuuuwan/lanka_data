import os
import tempfile

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lanka_data.api.how.map.FontUtils import FontUtils
from lanka_data.api.how.map.GeoDataUtils import GeoDataUtils
from lanka_data.api.how.map.LabelUtils import LabelUtils
from lanka_data.api.how.map.LegendUtils import LegendUtils
from lanka_data.api.how.map.RegionColorUtils import RegionColorUtils
from utils_future import Log

log = Log("PlotUtils")


class PlotUtils:
    DELIM_TITLE = " · "
    MAX_REGIONS_TO_LABEL = 100
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
    FONT_FAMILY = 'Fira Sans'

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
    def get_figure_specs(what, when, where, how):
        from lanka_data.api.what.WhatFactory import WhatFactory

        if '-' in when:
            when_parts = when.split('-')
            return {
                when_parts[0]: (
                    WhatFactory.from_what_and_when(what.title, when_parts[0]),
                    when_parts[0],
                    where,
                    how,
                ),
                when_parts[1]: (
                    WhatFactory.from_what_and_when(what.title, when_parts[1]),
                    when_parts[1],
                    where,
                    how,
                ),
                'Change': (what, when, where, how),
            }

        return {"": (what, when, where, how)}

    # flake8: noqa: CFQ002
    @staticmethod
    def plot_subfigure(
        figure_label,
        figure_spec,
        cmd,
        is_cartogram,
        subfig,
    ):
        what, when, where, how = figure_spec
        result_data = how.get_data(what, when, where)
        data_list = result_data["data_list"]
        n_regions = len(data_list)
        gdf_region = GeoDataUtils.get_geopandas_dataframe(
            data_list, is_cartogram
        ).copy()
        region_color_map, value_to_color = RegionColorUtils.get_color_spec(
            what, when, where, how
        ).unpack()
        gdf_region["color"] = gdf_region["region_id"].map(region_color_map)

        gs = subfig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
        ax = subfig.add_subplot(gs[0])
        legend_ax = subfig.add_subplot(gs[1])
        if n_regions > 400:
            edge_color, edge_width = "none", 0
        else:
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
            LabelUtils._draw_labels(gdf_region, ax)
        LegendUtils._draw_legend(value_to_color, ax, legend_ax)

        ax.set_axis_off()
        PlotUtils._plot_text(
            subfig,
            (0.5, 0.9),
            figure_label,
            fontsize=16,
            color="#000",
        )

    @staticmethod
    def plot_subfigures(what, when, where, how, cmd, is_cartogram):

        figure_specs = PlotUtils.get_figure_specs(what, when, where, how)

        n_figs = len(figure_specs)
        rows, cols = 1, n_figs
        fig = plt.figure(figsize=(PlotUtils.FIG_WIDTH, PlotUtils.FIG_HEIGHT))

        outer_gs = gridspec.GridSpec(rows, cols, figure=fig, top=1, bottom=0)
        subfigs_flat = [
            fig.add_subfigure(outer_gs[i, j])
            for i in range(rows)
            for j in range(cols)
        ]
        for (figure_label, figure_spec), subfig in zip(
            figure_specs.items(), subfigs_flat[:n_figs]
        ):

            PlotUtils.plot_subfigure(
                figure_label,
                figure_spec,
                cmd,
                is_cartogram,
                subfig,
            )

        return fig

    @staticmethod
    def _draw_header(fig, what, when, where, how):
        PlotUtils._plot_text(
            fig,
            (0.5, 0.975),
            f'{what.title} ({when}) - {where.get_description()} - {how.get_description()}',
            16,
            "#fff",
        )

    @staticmethod
    def _draw_footer(fig, cmd, source, source_url):
        PlotUtils._plot_text(
            fig,
            (0.5, 0.025),
            f"Data Source: {source} ({source_url})",
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
            facecolor='grey',
            edgecolor='none',
            zorder=0,
        )
        fig.patches.append(rect)
        rect = Rectangle(
            (0, 0.95),
            1,
            0.05,
            transform=fig.transFigure,
            facecolor='grey',
            edgecolor='none',
            zorder=0,
        )
        fig.patches.append(rect)

    @classmethod
    def draw_plot(cls, what, when, where, how, cmd, is_cartogram):
        FontUtils.install_font(cls.FONT_FAMILY)
        fig = PlotUtils.plot_subfigures(
            what, when, where, how, cmd, is_cartogram
        )

        region_data = how.get_data(what, when, where)
        source = region_data["source"]
        source_url = region_data["source_url"]
        PlotUtils._plot_rects(fig)
        PlotUtils._draw_header(fig, what, when, where, how)
        PlotUtils._draw_footer(fig, cmd, source, source_url)

        image_dir = os.path.join(PlotUtils.DIR_OUTPUT, cmd)
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, "Image.png")

        fig.savefig(image_path, dpi=200, bbox_inches=0)
        plt.close(fig)

        log.debug(f"Wrote {image_path}")
        return {
            "image_path": image_path,
            "source": source,
            "source_url": source_url,
        }
