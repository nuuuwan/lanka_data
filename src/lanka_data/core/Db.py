import logging
import os
import tempfile

import geopandas as gpd
import matplotlib.pyplot as plt
import requests

log = logging.getLogger(__name__)


class Db:
    def __init__(self, cmd: str):
        self.cmd = cmd

    @classmethod
    def get_temp_file_path_base(cls, cmd) -> str:
        tokens = cmd.lower().split("/")
        cmd_id = "-".join(tokens)
        temp_dir = os.path.join(tempfile.gettempdir(), "lanka_data")
        os.makedirs(temp_dir, exist_ok=True)
        return os.path.join(temp_dir, cmd_id)

    def run(self):
        tokens = self.cmd.split("/")
        token0_tokens = tokens[0].split(":")

        region_id = None
        parent_region_id = None
        region_type = None

        if len(token0_tokens) == 1:
            region_id = token0_tokens[0]
            region_type = self.get_region_type(region_id)
        elif len(token0_tokens) == 2:
            parent_region_id = token0_tokens[0]
            region_type = token0_tokens[1]

        n_tokens = len(tokens)
        if n_tokens == 1:
            return self.get_regions(region_type, region_id, parent_region_id)

        if n_tokens == 2:
            if tokens[1] == "JSON":
                return self.get_regions()
            if tokens[1] == "Map":
                return self.get_regions_map(
                    region_type,
                    region_id,
                    parent_region_id,
                    self.cmd,
                )

        raise ValueError(f"Invalid command: {self.cmd}")

    @classmethod
    def get_region_type(cls, region_id: str) -> str:
        region_type = None
        id_len = len(region_id)
        if region_id.startswith("LK"):
            region_type = {
                2: "country",
                4: "province",
                5: "district",
                7: "dsd",
                10: "gnd",
            }.get(id_len)

        if region_id.startswith("EC-"):
            region_type = {
                5: "ed",
                6: "pd",
            }.get(id_len)

        if region_type is not None:
            return region_type

        raise ValueError(f"Invalid region ID format: {region_id}")

    @classmethod
    def get_data_list_for_region_type(cls, region_type: str):

        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main"
            + f"/data/ents/{region_type}s.json"
        )
        try:
            log.debug(f"🌐 {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch data from {url}: {e}")

        data_list = response.json()
        return data_list

    @classmethod
    def get_regions(cls, region_type, region_id, parent_region_id):
        assert region_type
        assert (region_id or parent_region_id) and not (
            region_id and parent_region_id
        )

        regions = cls.get_data_list_for_region_type(region_type)

        if parent_region_id:
            regions = [d for d in regions if parent_region_id in d["id"]]
        elif region_id:
            regions = [d for d in regions if d["id"] == region_id]

        if not regions:
            raise ValueError("No regions found.")
        return regions

    @classmethod
    def get_regions_map(cls, region_type, region_id, parent_region_id, cmd):
        assert region_type
        assert (region_id or parent_region_id) and not (
            region_id and parent_region_id
        )

        precision_label = {"gnd": "e3_medium"}.get(region_type, "e4_large")

        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main"
            + f"/data/geo/topojson/{precision_label}/{region_type}s.topojson"
        )
        log.debug(f"🌐 {url}")
        gdf_region = gpd.read_file(url)

        if region_id:
            gdf_region = gdf_region[gdf_region["id"] == region_id]
        elif parent_region_id:
            gdf_region = gdf_region[
                gdf_region["id"].str.contains(parent_region_id)
            ]

        if gdf_region.empty:
            raise ValueError("No map data found.")

        n = len(gdf_region)
        cmap = plt.cm.tab20
        colors = [cmap(i % 20) for i in range(n)]

        gdf_region = gdf_region.copy()
        gdf_region["color"] = colors

        fig, ax = plt.subplots(figsize=(10, 8))
        gdf_region.plot(
            ax=ax,
            column="id",
            categorical=True,
            color=gdf_region["color"],
            edgecolor="white",
            linewidth=0.2,
        )

        if n <= 30:
            for _, row in gdf_region.iterrows():
                centroid = row.geometry.centroid
                ax.annotate(
                    row["id"],
                    xy=(centroid.x, centroid.y),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )

        ax.set_axis_off()

        image_path = cls.get_temp_file_path_base(cmd) + ".png"
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        log.info(f"Wrote {image_path}")
        plt.close(fig)
        os.system(f"open {image_path}")
        return {
            "image": image_path,
        }
