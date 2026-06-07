import os
import tempfile

import matplotlib.pyplot as plt

from lanka_data.how.GeoDataUtils import GeoDataUtils
from lanka_data.how.LabelUtils import LabelUtils
from lanka_data.how.LegendUtils import LegendUtils
from lanka_data.how.RegionColorUtils import RegionColorUtils
from utils_future import Log

log = Log("MapUtils")


class MapUtils:
    DELIM_TITLE = " · "
    MAX_REGIONS_TO_LABEL = 100
    DEFAULT_EDGE_COLOR = "#fff"
    DEFAULT_EDGE_WIDTH = 0.5

    @staticmethod
    def _render_figure(gdf_region, n_regions, value_to_color, title, source):
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        legend_ax = fig.add_subplot(gs[1])
        if n_regions > 100:
            edge_color, edge_width = "none", 0
        else:
            edge_color, edge_width = (
                MapUtils.DEFAULT_EDGE_COLOR,
                MapUtils.DEFAULT_EDGE_WIDTH,
            )

        gdf_region.plot(
            ax=ax,
            categorical=True,
            color=gdf_region["color"],
            edgecolor=edge_color,
            linewidth=edge_width,
        )
        if n_regions <= MapUtils.MAX_REGIONS_TO_LABEL:
            LabelUtils._draw_labels(gdf_region, ax)
        LegendUtils._draw_legend(value_to_color, ax, legend_ax)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        if source:
            fig.text(
                0.5,
                0.01,
                f"Source: {source}",
                ha="center",
                fontsize=7,
                color="gray",
            )
        return fig

    @staticmethod
    def draw_map(where, what, when, how):
        result_data = how.get_data(where, what, when)
        h = how.get_hash(where, what, when)
        data_list = result_data["data_list"]
        n_regions = len(data_list)
        gdf_region = GeoDataUtils.get_geopandas_dataframe(data_list).copy()
        region_color_map, value_to_color = (
            RegionColorUtils.get_region_color_map(result_data, how, what)
        )
        gdf_region["color"] = gdf_region["region_id"].map(region_color_map)
        title = MapUtils.DELIM_TITLE.join(
            how.get_title_items(where, what, when)
        )
        source = result_data.get("source", "")
        fig = MapUtils._render_figure(
            gdf_region, n_regions, value_to_color, title, source
        )
        image_dir = os.path.join(
            tempfile.gettempdir(), "lanka_data", "images"
        )
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"{h}.png")
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        log.debug(f"Wrote {image_path}")
        plt.close(fig)
        return {
            "image_path": image_path,
            "source": "Department of Census and Statistics, Sri Lanka",
            "source_url": "https://www.statistics.gov.lk/",
        }
