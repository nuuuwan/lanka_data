import random

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
    }

    @staticmethod
    def get_random_color():

        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    @staticmethod
    def get_colors_for_data_list(data_list):
        if data_list[0].get("values") is None:
            n_regions = len(data_list)
            cmap = plt.cm.tab20  # pylint: disable=no-member.
            colors = [cmap(i % 20) for i in range(n_regions)]
            return colors

        color_idx = {}
        colors = []
        for data in data_list:
            max_value_key = list(data["values"].keys())[0]
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
    def _draw_legend(data_list, colors, ax):
        if data_list[0].get("values") is not None:
            unique_colors = set(colors)
            for color in unique_colors:
                idx = colors.index(color)
                label = max(
                    data_list[idx]["values"],
                    key=data_list[idx]["values"].get,
                )
                ax.scatter([], [], color=color, label=label)
            ax.legend(fontsize=6)

    @staticmethod
    def draw_map(result, file_path_base: str, cmd: str):
        data_list = result["data_list"]
        region_ids = [d["region_id"] for d in data_list]
        n_regions = len(region_ids)
        gdf_region = RegionsGeoUtils.get_geopandas_dataframe(region_ids)

        if "by_party" in data_list[0]:
            data_list = [
                data | dict(values=data["by_party"]) for data in data_list
            ]

        gdf_region = gdf_region.copy()
        colors = RegionsMapUtils.get_colors_for_data_list(data_list)
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

        RegionsMapUtils._draw_legend(data_list, colors, ax)
        if cmd:
            ax.set_title(cmd, fontsize=10)
        ax.set_axis_off()

        image_path = f"{file_path_base}.png"
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        log.info(f"Wrote {image_path}")
        plt.close(fig)
        return {
            "image_path": image_path,
            "source": "Department of Census and Statistics, Sri Lanka",
            "source_url": "https://www.statistics.gov.lk/",
        }
