from lanka_data.region.RegionTypeUtils import RegionTypeUtils
from utils_future import Log
from utils_future.GeoUtils import GeoUtils
from utils_future.timer import timer

log = Log("RegionParserMixin")


class RegionParserMixin:

    @classmethod
    def get_region_ids_from_range(cls, from_region_id, to_region_id):
        region_type = RegionTypeUtils.get_region_type(from_region_id)
        if region_type != RegionTypeUtils.get_region_type(to_region_id):
            raise ValueError(
                f"Region types do not match: {from_region_id}, {to_region_id}"
            )
        region_year = cls._get_region_year(from_region_id)
        if region_year != cls._get_region_year(to_region_id):
            raise ValueError(
                f"Region years do not match: {from_region_id}, {to_region_id}"
            )

        raw_regions = cls._get_raw_region_data_list_for_region_type(
            region_type, region_year
        )
        region_ids = [
            r["region_id"]
            for r in raw_regions
            if from_region_id <= r["region_id"] <= to_region_id
        ]
        if not region_ids:
            raise ValueError(
                f"No regions found in range: {from_region_id}...{to_region_id}"
            )
        return region_ids

    @classmethod
    def _is_within_radius(cls, radius_km, center_region, other_region):
        distance = GeoUtils.haversine_distance(
            center_region["center_lat"],
            center_region["center_lng"],
            other_region["center_lat"],
            other_region["center_lng"],
        )
        return distance <= radius_km

    @classmethod
    def get_region_ids_from_region_radius(cls, region_id, radius_km):
        region_type = RegionTypeUtils.get_region_type(region_id)
        region_year = cls._get_region_year(region_id)
        raw_regions = cls._get_raw_region_data_list_for_region_type(
            region_type, region_year
        )

        center_region = None
        for region in raw_regions:
            if region["region_id"] == region_id:
                center_region = region
                break

        if not center_region:
            raise ValueError(f"Region ID not found: {region_id}")

        nearby_region_ids = [
            r["region_id"]
            for r in raw_regions
            if cls._is_within_radius(radius_km, center_region, r)
        ]
        if not nearby_region_ids:
            raise ValueError(
                f"No regions found within {radius_km} km of {region_id}"
            )
        return nearby_region_ids

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
