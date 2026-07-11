from lanka_data.api.fields.Where import Where
from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils
from lanka_data.datasets.region.rivers.RiversData import RiversData
from utils_future import timer

from .RegionParserRadiusMixin import RegionParserRadiusMixin


class RegionParserMixin(RegionParserRadiusMixin):
    RIVERS_REGION_TYPE = "rivers"

    @classmethod
    def _is_rivers_region_id(cls, region_id):
        return (
            RegionTypeUtils.get_region_type(region_id)
            == cls.RIVERS_REGION_TYPE
        )

    @classmethod
    def _filter_river_regions(cls, region_ids):
        region_idx = {
            region["region_id"]: region
            for region in RiversData.get_river_regions()
        }
        regions = []
        for region_id in region_ids:
            region = region_idx.get(region_id)
            if region is None:
                raise ValueError(f"River region not found: {region_id}")
            regions.append(region)
        return regions

    @classmethod
    def parse_parent_part(cls, parent_part: str):
        if "..." in parent_part:
            from_region_id, to_region_id = parent_part.split("...")
            parent_region_ids = cls.get_region_ids_from_range(
                from_region_id, to_region_id
            )
            region_year = cls._get_region_year(parent_region_ids[0])
            return parent_region_ids, region_year

        if "@" in parent_part:
            region_id, radius_km_str = parent_part.split("@")
            radius_km = float(radius_km_str)
            parent_region_ids = cls.get_region_ids_from_region_radius(
                region_id, radius_km
            )
            region_year = cls._get_region_year(region_id)
            return parent_region_ids, region_year

        parent_region_ids = parent_part.split(",")
        region_year = cls._get_region_year(parent_region_ids[0])
        parent_region_ids = [
            parent_region_id.split("-pre")[0]
            for parent_region_id in parent_region_ids
        ]
        return parent_region_ids, region_year

    @classmethod
    def _get_regions_without_child_type(cls, parent_region_ids):
        if all(cls._is_rivers_region_id(r) for r in parent_region_ids):
            return cls._filter_river_regions(parent_region_ids)
        return cls._get_raw_region_data_list_for_region_ids(parent_region_ids)

    @classmethod
    @timer
    def get_raw_regions(
        cls, parent_region_ids, child_region_type, region_year
    ):
        if child_region_type == cls.RIVERS_REGION_TYPE:
            return RiversData.get_river_regions(), region_year
        if not child_region_type:
            return (
                cls._get_regions_without_child_type(parent_region_ids),
                region_year,
            )
        child_raw_regions = cls._get_raw_region_data_list_for_region_type(
            child_region_type, region_year
        )
        filtered_child_regions = [
            r
            for r in child_raw_regions
            if cls.has_some_parent(r["region_id"], parent_region_ids)
        ]
        return filtered_child_regions, region_year

    @classmethod
    def parse(cls, token: str):
        token = Where.strip_top(token)
        if ":" in token:
            parent_part, child_region_type = token.split(":")
        else:
            parent_part = token
            child_region_type = None
        parent_region_ids, region_year = cls.parse_parent_part(parent_part)
        regions, region_year = cls.get_raw_regions(
            parent_region_ids, child_region_type, region_year
        )
        return regions, region_year
