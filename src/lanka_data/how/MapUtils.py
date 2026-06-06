import colorsys
import os
import random
import tempfile

import matplotlib.pyplot as plt

from lanka_data.how.GeoDataUtils import GeoDataUtils
from utils_future import Log

log = Log("MapUtils")

random.seed(0)


class COLOR:
    GOLD = "#FFBE29"
    DARK_ORANGE = "#EB7400"
    TEAL = "#00534E"
    BLUE = "#0000FF"
    MAROON = "#800080"
    GRAY = "#808080"
    DARK_RED = "#8D153A"
    GREEN = "#008000"
    RED = "#FF0000"
    ORANGE = "#FFA500"


class MapUtils:
    DELIM_TITLE = " · "
    MAX_REGIONS_TO_LABEL = 100
    MAX_LEGEND_ITEMS = 7
    COLOR_IDX = {
        # Religion
        "buddhist": COLOR.GOLD,
        "hindu": COLOR.DARK_ORANGE,
        "islam": COLOR.TEAL,
        "other_christian": COLOR.BLUE,
        "roman_catholic": COLOR.MAROON,
        "other": COLOR.GRAY,
        # Ethnicity
        "sinhalese": COLOR.DARK_RED,
        "sl_tamil": COLOR.DARK_ORANGE,
        "sri_lanka_tamil": COLOR.DARK_ORANGE,
        "ind_tamil": COLOR.BLUE,
        "indian_tamil_or_malaiyaga_thamilar": COLOR.BLUE,
        "sl_moor": COLOR.TEAL,
        "sri_lanka_moor_or_muslim": COLOR.TEAL,
        "malay": COLOR.GREEN,
        # Political Party
        "SLPP": COLOR.DARK_RED,
        "UPFA": COLOR.BLUE,
        "PA": COLOR.BLUE,
        "SLFP": COLOR.BLUE,
        "NPP": COLOR.RED,
        "SJB": COLOR.GREEN,
        "UNP": COLOR.GREEN,
        "NDF": COLOR.GREEN,
        "IND9": COLOR.ORANGE,
        "SLMP": COLOR.MAROON,
        "ACTC": COLOR.ORANGE,
        "ITAK": COLOR.ORANGE,
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
        value_to_rank = {
            value: rank for rank, value in enumerate(sorted(set(pct_values)))
        }
        n = len(value_to_rank)
        for data in data_list:
            value = data["pct_values"][how.params]
            rank = value_to_rank[value]
            p = rank / (n - 1)
            hue = (1 - p) * 0.67
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
    def _is_light_color(color):
        if isinstance(color, str):
            color = color.lstrip("#")
            if len(color) == 6:
                r, g, b = (
                    int(color[0:2], 16) / 255,
                    int(color[2:4], 16) / 255,
                    int(color[4:6], 16) / 255,
                )
            else:
                return False
        else:
            # assume RGBA or RGB tuple
            r, g, b = color[0], color[1], color[2]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance > 0.5

    @staticmethod
    def _draw_labels(gdf_region, ax):
        for _, row in gdf_region.iterrows():
            centroid = row.geometry.centroid
            bg_color = row.get("color", "black")
            text_color = (
                "black" if MapUtils._is_light_color(bg_color) else "white"
            )
            ax.annotate(
                row["name"],
                xy=(centroid.x, centroid.y),
                ha="center",
                va="center",
                fontsize=5,
                color=text_color,
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
            n_actual = len(value_and_color)
            n_required = MapUtils.MAX_LEGEND_ITEMS - 1

            new_value_and_color = []
            for i in range(n_required):
                idx = int(i * n_actual / n_required)
                new_value_and_color.append(value_and_color[idx])
            new_value_and_color.append(value_and_color[-1])  # add max value
            value_and_color = new_value_and_color

        for value, color in value_and_color:
            ax.scatter(
                [],
                [],
                color=color,
                label=MapUtils._format_legend_label(value),
            )
        ax.legend(fontsize=6)

    @staticmethod
    def draw_map(where, what, when, how):
        result_data = how.get_data(where, what, when)
        h = how.get_hash(where, what, when)

        data_list = result_data["data_list"]
        region_ids = [d["region_id"] for d in data_list]
        n_regions = len(region_ids)
        gdf_region = GeoDataUtils.get_geopandas_dataframe(region_ids)

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

        image_dir = os.path.join(
            tempfile.gettempdir(), "lanka_data", "images"
        )
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
