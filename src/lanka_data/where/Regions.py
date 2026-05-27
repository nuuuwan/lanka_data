import logging
from functools import cached_property

from lanka_data.where.Where import Where
from utils_future import WWW

log = logging.getLogger(__name__)


class Regions(Where):

    def __init__(self, regions: list[str]):
        self.regions = regions

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
        return WWW(url).read_json()

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
