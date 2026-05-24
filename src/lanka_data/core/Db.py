import logging
import os
import tempfile
from functools import cache, cached_property

import geopandas as gpd
import matplotlib.pyplot as plt
import requests

log = logging.getLogger(__name__)


class Db:
    def __init__(self, cmd: str):
        self.cmd = cmd

    @cached_property
    def cmd_id(self) -> str:
        tokens = self.cmd.lower().split("/")
        return "-".join(tokens)

    @cached_property
    def temp_dir(self) -> str:
        temp_dir = os.path.join(tempfile.gettempdir(), "lanka_data")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    @cached_property
    def temp_file_path_base(self) -> str:
        return os.path.join(
            self.temp_dir,
            self.cmd_id,
        )

    def run(self):
        tokens = self.cmd.split("/")
        self.region_id = tokens[0]
        n_tokens = len(tokens)
        if n_tokens == 1:
            return self.region

        if n_tokens == 2:
            if tokens[1] == "JSON":
                return self.region
            if tokens[1] == "Map":
                return self.region_map

        raise ValueError(f"Invalid command: {self.cmd}")

    @cached_property
    def region_type(self) -> str:
        region_type = None
        id_len = len(self.region_id)
        if self.region_id.startswith("LK"):
            region_type = {
                2: "country",
                4: "province",
                5: "district",
                7: "dsd",
                10: "gnd",
            }.get(id_len)

        if self.region_id.startswith("EC-"):
            region_type = {
                5: "ed",
                6: "pd",
            }.get(id_len)

        if region_type is not None:
            return region_type

        raise ValueError(f"Invalid region ID format: {self.region_id}")

    @classmethod
    @cache
    def get_data_idx_for_region_type(cls, region_type: str):

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

        data = response.json()
        data_idx = {d["id"]: d for d in data}
        return data_idx

    @cached_property
    def region(self):
        region_type = self.region_type

        data_idx = self.get_data_idx_for_region_type(region_type)
        if self.region_id not in data_idx:
            raise ValueError(f"Region ID {self.region_id} not found in data")
        return dict(region_type=region_type) | data_idx[self.region_id]

    @cached_property
    def region_map(self):

        region_type = self.region_type
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main"
            + f"/data/geo/topojson/e4_large/{region_type}s.topojson"
        )
        log.debug(f"🌐 {url}")
        gdf = gpd.read_file(url)
        gdf_region = gdf[gdf["id"] == self.region_id]
        if gdf_region.empty:
            raise ValueError(
                f"Region ID {self.region_id} not found in map data"
            )

        fig, ax = plt.subplots(figsize=(10, 8))
        gdf_region.plot(
            ax=ax, edgecolor="black", linewidth=0.5, facecolor="#cce5df"
        )
        ax.set_axis_off()

        image_path = self.temp_file_path_base + ".png"
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        log.info(f"Wrote {image_path}")
        plt.close(fig)
        os.system(f"open {image_path}")
        return {
            "image": image_path,
        }
