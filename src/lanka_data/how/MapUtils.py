import colorsys
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
    MAX_LEGEND_ITEMS = 7
    COLOR_IDX = {
        # Religion
        "buddhist": "#FFBE29",
        "hindu": "#EB7400",
        "islam": "#00534E",
        "other_christian": "blue",
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
        region_color_map = {
            data["region_id"]: cmap(i % 20)
            for i, data in enumerate(data_list[:n_regions])
        }
        value_to_color = None
        return region_color_map, value_to_color

    @staticmethod
    def get_func_key_getter(how, what):
        param = how.params
        if not param or param == "Top":
            idx = 0
        elif param == "2nd":
            idx = 1
        elif param == "3rd":
            idx = 2
        elif param == "Bottom":
            idx = -1
        else:
            return None

        def func_key_getter(data):
            return list(what.get_values(data).keys())[idx]

        return func_key_getter

    @staticmethod
    def get_colors_for_data_list_with_values_key(result_data, how, what):
        data_list = result_data["data_list"]
        value_to_color = {}
        region_color_map = {}
        pct_values = [data["pct_values"][how.params] for data in data_list]
        min_value = min(pct_values)
        max_value = max(pct_values)
        for data in data_list:
            value = data["pct_values"][how.params]
            p = (value - min_value) / (max_value - min_value)
            hue = (1 - p) * 0.6
            sat = 1.0
            light = 0.5
            color = colorsys.hls_to_rgb(hue, light, sat)
            value_to_color[value] = color
            region_color_map[data["region_id"]] = color
        return region_color_map, value_to_color

    @staticmethod
    def get_colors_for_data_list_with_values_order(
        result_data, how, what, func_key_getter
    ):
        data_list = result_data["data_list"]
        value_to_color = {}
        region_color_map = {}
        for data in data_list:
            key = func_key_getter(data) if func_key_getter else None
            if key not in value_to_color:
                color = (
                    MapUtils.COLOR_IDX[key]
                    if key in MapUtils.COLOR_IDX
                    else MapUtils.get_random_color()
                )
                value_to_color[key] = color
            region_color_map[data["region_id"]] = value_to_color[key]
        return region_color_map, value_to_color

    @staticmethod
    def get_colors_for_data_list_with_values(result_data, how, what):
        func_key_getter = MapUtils.get_func_key_getter(how, what)
        if func_key_getter:
            return MapUtils.get_colors_for_data_list_with_values_order(
                result_data, how, what, func_key_getter
            )
        return MapUtils.get_colors_for_data_list_with_values_key(
            result_data,
            how,
            what,
        )

    @staticmethod
    def get_colors_for_data_list(result_data, how, what):
        data_list = result_data["data_list"]
        if what.get_values(data_list[0]) is None:
            return MapUtils.get_colors_for_data_list_without_values(
                result_data
            )
        return MapUtils.get_colors_for_data_list_with_values(
            result_data, how, what
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
    def _format_legend_label(value):
        if isinstance(value, (int, float)):
            return f"{value:.1%}"
        return str(value)

    @staticmethod
    def _draw_legend(value_to_color, ax):
        if value_to_color is None:
            return
        value_and_color = sorted(value_to_color.items())
        if len(value_and_color) > MapUtils.MAX_LEGEND_ITEMS:
            value_and_color = (
                value_and_color[: MapUtils.MAX_LEGEND_ITEMS // 2]
                + value_and_color[-(MapUtils.MAX_LEGEND_ITEMS // 2) :]
            )
        for value, color in value_and_color:
            ax.scatter(
                [], [], color=color, label=MapUtils._format_legend_label(value)
            )
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
        region_color_map, value_to_color = MapUtils.get_colors_for_data_list(
            result_data, how, what
        )
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

        MapUtils._draw_legend(value_to_color, ax)
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
