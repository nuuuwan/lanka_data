import hashlib
import os
import random
import tempfile

import matplotlib.pyplot as plt

from lanka_data.where.RegionsGeoUtils import RegionsGeoUtils
from utils_future import Log

log = Log("RegionsMapUtils")


class RegionsMapUtils:
    MAX_REGIONS_TO_LABEL = 100
    COLOR_IDX = {
        # Religion
        "buddhist": "#FFBE29",
        "hindu": "#EB7400",
        "islam": "#00534E",
        "christian": "blue",
        "roman_catholic": "purple",
        "other": "gray",
        # Ethnicity
        "sinhalese": "#8D153A",
        "sl_tamil": "#EB7400",
        "ind_tamil": "blue",
        "sl_moor": "#00534E",
        "malay": "green",
        # Political Party
        "SLPP": "#8D153A",
        "UPFA": "blue",
        "PA": "blue",
        "SLFP": "blue",
        "NPP": "red",
        "SJB": "green",
        "UNP": "green",
        "NDF": "green",
        "IND9": "orange",
        "SLMP": "purple",
        "ACTC": "orange",
        "ITAK": "orange",
    }

    @staticmethod
    def get_random_color():

        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    @staticmethod
    def get_colors_for_data_list(result, data_list):
        if result.what.get_values(data_list[0]) is None:
            n_regions = len(data_list)
            cmap = plt.cm.tab20  # pylint: disable=no-member.
            colors = [cmap(i % 20) for i in range(n_regions)]
            return colors

        color_idx = {}
        colors = []
        for data in data_list:
            max_value_key = list(result.what.get_values(data).keys())[0]
            if max_value_key not in color_idx:
                color = (
                    RegionsMapUtils.COLOR_IDX[max_value_key]
                    if max_value_key in RegionsMapUtils.COLOR_IDX
                    else RegionsMapUtils.get_random_color()
                )
                color_idx[max_value_key] = color
            colors.append(color_idx[max_value_key])
        return colors

    @staticmethod
    def _draw_labels(gdf_region, ax):
        for _, row in gdf_region.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(
                row["name"],
                xy=(centroid.x, centroid.y),
                ha="center",
                va="center",
                fontsize=5,
                color="white",
            )

    @staticmethod
    def _draw_legend(result, data_list, colors, ax):

        if result.what.get_values(data_list[0]) is not None:
            unique_colors = set(colors)
            for color in unique_colors:
                idx = colors.index(color)
                label = max(
                    result.what.get_values(data_list[idx]),
                    key=result.what.get_values(data_list[idx]).get,
                )
                ax.scatter([], [], color=color, label=label)
            ax.legend(fontsize=6)

    @staticmethod
    def draw_map(result, title: str):
        result_data = result.get_data()
        h = hashlib.md5(str(result_data).encode("utf-8")).hexdigest()[:8]

        data_list = result_data["data_list"]
        region_ids = [d["region_id"] for d in data_list]
        n_regions = len(region_ids)
        gdf_region = RegionsGeoUtils.get_geopandas_dataframe(region_ids)

        gdf_region = gdf_region.copy()
        colors = RegionsMapUtils.get_colors_for_data_list(result, data_list)
        gdf_region["color"] = colors

        fig, ax = plt.subplots(figsize=(10, 8))
        gdf_region.plot(
            ax=ax,
            categorical=True,
            color=gdf_region["color"],
            edgecolor="white",
            linewidth=0.2,
        )

        if n_regions <= RegionsMapUtils.MAX_REGIONS_TO_LABEL:
            RegionsMapUtils._draw_labels(gdf_region, ax)

        RegionsMapUtils._draw_legend(result, data_list, colors, ax)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

        source = result_data.get("source", "")
        if source:
            fig.text(
                0.5,
                0.01,
                f"Source: {source}",
                ha="center",
                fontsize=7,
                color="gray",
            )

        image_dir = os.path.join(tempfile.gettempdir(), "lanka_data", "images")
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"{h}.png")
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        log.info(f"Wrote {image_path}")
        plt.close(fig)
        return {
            "image_path": image_path,
            "source": "Department of Census and Statistics, Sri Lanka",
            "source_url": "https://www.statistics.gov.lk/",
        }
