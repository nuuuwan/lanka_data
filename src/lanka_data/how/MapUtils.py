import hashlib
import os
import random
import tempfile

import matplotlib.pyplot as plt

from lanka_data.how.GeoUtils import GeoUtils
from utils_future import Log

log = Log("MapUtils")


class MapUtils:
    DELIM_TITLE = " · "
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
    def get_colors_for_data_list_without_values(result_data):
        data_list = result_data["data_list"]
        n_regions = len(data_list)
        cmap = plt.cm.tab20  # pylint: disable=no-member.
        return {
            data["region_id"]: cmap(i % 20)
            for i, data in enumerate(data_list[:n_regions])
        }

    @staticmethod
    def get_colors_for_data_list_with_values(result_data, what):
        data_list = result_data["data_list"]
        value_to_color = {}
        region_color_map = {}
        for data in data_list:
            max_value_key = list(what.get_values(data).keys())[0]
            if max_value_key not in value_to_color:
                color = (
                    MapUtils.COLOR_IDX[max_value_key]
                    if max_value_key in MapUtils.COLOR_IDX
                    else MapUtils.get_random_color()
                )
                value_to_color[max_value_key] = color
            region_color_map[data["region_id"]] = value_to_color[max_value_key]
        return region_color_map

    @staticmethod
    def get_colors_for_data_list(result_data, what):
        data_list = result_data["data_list"]
        if what.get_values(data_list[0]) is None:
            return MapUtils.get_colors_for_data_list_without_values(
                result_data
            )
        return MapUtils.get_colors_for_data_list_with_values(
            result_data, what
        )  # region_id -> color

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
    def _draw_legend(result_data, what, data_list, region_color_map, ax):
        if what.get_values(data_list[0]) is not None:
            seen = {}
            for data in data_list:
                color = region_color_map[data["region_id"]]
                if color not in seen:
                    label = max(
                        what.get_values(data),
                        key=what.get_values(data).get,
                    )
                    seen[color] = label
            for color, label in sorted(seen.items()):
                ax.scatter([], [], color=color, label=label)
            ax.legend(fontsize=6)

    @staticmethod
    def draw_map(where, what, when, how):
        result_data = how.get_data(where, what, when)
        h = hashlib.md5(str(result_data).encode("utf-8")).hexdigest()[:8]

        data_list = result_data["data_list"]
        region_ids = [d["region_id"] for d in data_list]
        n_regions = len(region_ids)
        gdf_region = GeoUtils.get_geopandas_dataframe(region_ids)

        gdf_region = gdf_region.copy()
        region_color_map = MapUtils.get_colors_for_data_list(result_data, what)
        gdf_region["color"] = gdf_region["id"].map(region_color_map)

        fig, ax = plt.subplots(figsize=(10, 8))
        gdf_region.plot(
            ax=ax,
            categorical=True,
            color=gdf_region["color"],
            edgecolor="white",
            linewidth=0.2,
        )

        if n_regions <= MapUtils.MAX_REGIONS_TO_LABEL:
            MapUtils._draw_labels(gdf_region, ax)

        MapUtils._draw_legend(
            result_data, what, data_list, region_color_map, ax
        )
        title = MapUtils.DELIM_TITLE.join(
            how.get_title_items(where, what, when)
        )
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
