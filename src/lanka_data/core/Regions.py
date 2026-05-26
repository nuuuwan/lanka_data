import logging
from functools import cache, cached_property

import geopandas
import matplotlib.pyplot as plt
import requests

from lanka_data.core.Where import Where

log = logging.getLogger(__name__)


class Regions(Where):
    MAX_REGIONS_TO_LABEL = 100

    def __init__(self, regions: list[str]):
        self.regions = regions

    @classmethod
    @cache
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

    @cached_property
    def region_type(self):
        return self.get_region_type(self.regions[0]["id"])

    @classmethod
    def _get_data_list_for_region_type(cls, region_type: str):

        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main"
            + f"/data/ents/{region_type}s.json"
        )
        log.debug(f"🌐 {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data_list = response.json()
        return data_list

    @classmethod
    def from_region_id(cls, region_id):
        region_type = cls.get_region_type(region_id)
        regions = cls._get_data_list_for_region_type(region_type)
        regions = [d for d in regions if d["id"] == region_id]
        if not regions:
            raise ValueError(f"Region ID not found: {region_id}")
        return cls(regions)

    @classmethod
    def is_parent(cls, region, parent_region_id) -> bool:
        if parent_region_id == "LK":
            return True

        region_id = region["id"]
        if parent_region_id in region_id:
            return True

        parent_region_type = cls.get_region_type(parent_region_id)
        parent_region_id_key = f"{parent_region_type}_id"
        if region.get(parent_region_id_key) == parent_region_id:
            return True

        return False

    @classmethod
    def from_parent_region_id_and_region_type(
        cls, region_type, parent_region_id
    ):
        regions = cls._get_data_list_for_region_type(region_type)
        regions = [
            region
            for region in regions
            if cls.is_parent(region, parent_region_id)
        ]
        if not regions:
            raise ValueError(
                f"No regions found for parent ID: {parent_region_id}"
            )
        return cls(regions)

    def _get_geopandas_dataframe(self):
        region_type = self.region_type
        precision_label = {"gnd": "e3_medium"}.get(region_type, "e4_large")
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main"
            + f"/data/geo/topojson/{precision_label}/{region_type}s.topojson"
        )
        log.debug(f"🌐 {url}")
        gdf_region = geopandas.read_file(url)

        region_ids = [d["id"] for d in self.regions]
        gdf_region = gdf_region[gdf_region["id"].isin(region_ids)]

        if gdf_region.empty:
            raise ValueError("No map data found.")
        return gdf_region

    def draw_map(self, file_path_base: str):
        gdf_region = self._get_geopandas_dataframe()
        n_regions = len(gdf_region)
        cmap = plt.cm.tab20  # pylint: disable=no-member.
        colors = [cmap(i % 20) for i in range(n_regions)]
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

        if n_regions <= self.MAX_REGIONS_TO_LABEL:
            for _, row in gdf_region.iterrows():
                centroid = row.geometry.centroid
                ax.annotate(
                    row["id"],
                    xy=(centroid.x, centroid.y),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                )
        ax.set_axis_off()

        image_path = f"{file_path_base}.png"
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        log.info(f"Wrote {image_path}")
        plt.close(fig)
        return {
            "image_path": image_path,
        }
