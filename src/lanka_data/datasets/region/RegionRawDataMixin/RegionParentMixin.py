from functools import cache

from lanka_data.datasets.region.rivers.RiversData import RiversData
from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils


class RegionParentMixin:
    RIVERS_REGION_TYPE = "rivers"

    @classmethod
    @cache
    def _river_name(cls, region_id):
        for region in RiversData.get_river_regions():
            if region["region_id"] == region_id:
                return region["region_name"]
        raise ValueError(f"River region not found: {region_id}")

    @classmethod
    @cache
    def is_parent(cls, parent_region_id: str, child_region_id: str) -> bool:
        parent_region_id = parent_region_id.split("-pre")[0]
        if parent_region_id == "LK" or parent_region_id in child_region_id:
            return True
        parent_region_type = RegionTypeUtils.get_region_type(parent_region_id)
        child_raw_region = cls._get_raw_region_data_for_region_id(
            child_region_id
        )
        parent_id_key = f"{parent_region_type}_id"
        return (
            parent_id_key in child_raw_region
            and child_raw_region[parent_id_key] == parent_region_id
        )

    @classmethod
    def has_some_parent(cls, region_id, parent_region_ids):
        for parent_region_id in parent_region_ids:
            if cls.is_parent(parent_region_id, region_id):
                return True
        return False

    @classmethod
    @cache
    def get_full_name(cls, region_id):
        region_type = RegionTypeUtils.get_region_type(region_id)
        if region_type == cls.RIVERS_REGION_TYPE:
            return cls._river_name(region_id)
        raw_data = cls._get_raw_region_data_for_region_id(region_id)
        region_name = raw_data["region_name"]
        if region_type == "country":
            return region_name
        return f"the {region_name} {region_type.upper()}"
