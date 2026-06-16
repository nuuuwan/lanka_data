import os
import tempfile

import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from lanka_data.api.how.map.GeoDataUtils import GeoDataUtils
from lanka_data.api.how.map.LabelUtils import LabelUtils
from lanka_data.api.how.map.LegendUtils import LegendUtils
from lanka_data.api.how.map.RegionColorUtils import RegionColorUtils
from utils_future import Log

log = Log("PlotUtils")

_ubuntu_font = next(
    (f.fname for f in fm.fontManager.ttflist if "Ubuntu" in f.name),
    None,
)
if _ubuntu_font:
    plt.rcParams["font.family"] = "Ubuntu"
else:
    log.warning("Ubuntu font not found; using default font.")


class PlotUtils:
    DELIM_TITLE = " · "
    MAX_REGIONS_TO_LABEL = 100
    DEFAULT_EDGE_COLOR = "#fff"
    DEFAULT_EDGE_WIDTH = 0.2
    DIR_OUTPUT = os.path.join(
        tempfile.gettempdir(),
        "lanka_data",
        "output",
    )

    @staticmethod
    def get_figure_specs(where, what, when, how):
        from lanka_data.api.how.HowFactory import HowFactory
        from lanka_data.api.what.WhatFactory import WhatFactory

        if '-' in when:
            when_parts = when.split('-')
            if how.params:
                how_label = how.how_label.split(":")[0]
                how_without_params = HowFactory.from_how_cmd(how_label)
            return {
                when_parts[1]: (
                    where,
                    WhatFactory.from_what_and_when(what.title, when_parts[1]),
                    when_parts[1],
                    how_without_params,
                ),
                when_parts[0]: (
                    where,
                    WhatFactory.from_what_and_when(what.title, when_parts[0]),
                    when_parts[0],
                    how_without_params,
                ),
                'diff': (where, what, when, how),
            }

        return {"Fig": (where, what, when, how)}

    # flake8: noqa: CFQ002
    @staticmethod
    def plot_figure(
        figure_label,
        figure_spec,
        cmd,
        is_cartogram,
        subfig,
    ):
        where, what, when, how = figure_spec
        print(figure_spec)
        result_data = how.get_data(where, what, when)
        data_list = result_data["data_list"]
        n_regions = len(data_list)
        gdf_region = GeoDataUtils.get_geopandas_dataframe(
            data_list, is_cartogram
        ).copy()
        region_color_map, value_to_color = (
            RegionColorUtils.get_region_color_map(result_data, how, what)
        )
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
        subfig.text(
            0.5,
            0.85,
            figure_label,
            transform=subfig.transSubfigure,
            ha="center",
            va="bottom",
            fontsize=10,
            color="gray",
        )

    @staticmethod
    def plot_figures(where, what, when, how, cmd, is_cartogram):

        figure_specs = PlotUtils.get_figure_specs(where, what, when, how)

        n_figs = len(figure_specs)
        rows, cols = 1, n_figs
        fig = plt.figure(figsize=(8 * cols, 8 * rows))

        outer_gs = gridspec.GridSpec(
            rows, cols, figure=fig, top=0.85, bottom=0.15
        )
        subfigs_flat = [
            fig.add_subfigure(outer_gs[i, j])
            for i in range(rows)
            for j in range(cols)
        ]
        for (figure_label, figure_spec), subfig in zip(
            figure_specs.items(), subfigs_flat[:n_figs]
        ):

            PlotUtils.plot_figure(
                figure_label,
                figure_spec,
                cmd,
                is_cartogram,
                subfig,
            )

        return fig

    @classmethod
    def draw_plot(cls, where, what, when, how, cmd, is_cartogram):

        fig = PlotUtils.plot_figures(
            where, what, when, how, cmd, is_cartogram
        )

        image_dir = os.path.join(PlotUtils.DIR_OUTPUT, cmd)
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, "Image.png")

        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        log.debug(f"Wrote {image_path}")
        return {
            "image_path": image_path,
            "source": "Department of Census and Statistics, Sri Lanka",
            "source_url": "https://www.statistics.gov.lk/",
        }
