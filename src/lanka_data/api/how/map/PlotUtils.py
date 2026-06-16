import os
import tempfile

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

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

    # flake8: noqa: CFQ002
    @staticmethod
    def _render_figure(
        gdf_region,
        n_regions,
        value_to_color,
        how_description,
        what_description,
        when_description,
        where_description,
        source,
        cmd,
    ):
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        legend_ax = fig.add_subplot(gs[1])
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

        ax.set_title("")  # clear default title space
        # Centre x over the visible content (map ax only, or map+legend)
        ax_pos = ax.get_position()
        right_x = (
            legend_ax.get_position().x1
            if legend_ax.get_visible()
            else ax_pos.x1
        )
        center_x = (ax_pos.x0 + right_x) / 2
        t = fig.transFigure

        fig.text(
            center_x,
            0.97,
            (
                f"{what_description} ({when_description})"
                if what_description != "Basic Information"
                else "Geo Boundaries"
            ),
            transform=t,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color="black",
        )
        fig.text(
            center_x,
            0.92,
            where_description,
            transform=t,
            ha="center",
            va="bottom",
            fontsize=14,
            color="black",
        )

        fig.text(
            center_x,
            0.87,
            how_description,
            transform=t,
            ha="center",
            va="bottom",
            fontsize=12,
            color="grey",
        )

        ax.set_axis_off()
        if source:
            fig.text(
                center_x,
                0.08,
                f"Source: {source}",
                transform=t,
                ha="center",
                fontsize=10,
                color="darkgray",
            )
        if cmd:
            fig.text(
                center_x,
                0.04,
                "Command: /" + cmd,
                transform=t,
                ha="center",
                fontsize=7,
                color="gray",
            )

        return fig

    @classmethod
    def draw_plot(cls, where, what, when, how, cmd, is_cartogram):
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

        where_description = where.get_description()
        what_description = what.get_description()
        when_description = when
        how_description = how.get_description()

        source = result_data.get("source", "")
        fig = PlotUtils._render_figure(
            gdf_region,
            n_regions,
            value_to_color,
            how_description,
            what_description,
            when_description,
            where_description,
            source,
            cmd,
        )
        image_dir = os.path.join(PlotUtils.DIR_OUTPUT, cmd)
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, "Image.png")

        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        with Image.open(image_path) as img:
            bordered = ImageOps.expand(img, border=2, fill="#404040")
            bordered.save(image_path)
        log.debug(f"Wrote {image_path}")
        return {
            "image_path": image_path,
            "source": "Department of Census and Statistics, Sri Lanka",
            "source_url": "https://www.statistics.gov.lk/",
        }
