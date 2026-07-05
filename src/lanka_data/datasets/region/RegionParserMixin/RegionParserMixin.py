from utils_future import timer

from .RegionParserRadiusMixin import RegionParserRadiusMixin


class RegionParserMixin(RegionParserRadiusMixin):
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
            region_id, _ = parent_part.split("@")
            parent_region_ids = [region_id]
            region_year = cls._get_region_year(parent_region_ids[0])
            return parent_region_ids, region_year

        parent_region_ids = parent_part.split(",")
        region_year = cls._get_region_year(parent_region_ids[0])
        parent_region_ids = [
            parent_region_id.split("-pre")[0]
            for parent_region_id in parent_region_ids
        ]
        return parent_region_ids, region_year

    @classmethod
    @timer
    def get_raw_regions(
        cls, parent_region_ids, child_region_type, region_year
    ):
        if not child_region_type:
            return (
                cls._get_raw_region_data_list_for_region_ids(
                    parent_region_ids
                ),
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
