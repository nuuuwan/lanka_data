from api.utils_future.GeoUtils import GeoUtils


class RegionParserRadiusMixin:
    @classmethod
    def get_region_ids_from_range(cls, from_region_id, to_region_id):
        from datasets.region.RegionTypeUtils import RegionTypeUtils

        region_type = RegionTypeUtils.get_region_type(from_region_id)
        if region_type != RegionTypeUtils.get_region_type(to_region_id):
            raise ValueError(
                f"Region types do not match: "
                f"{from_region_id}, {to_region_id}"
            )
        region_year = cls._get_region_year(from_region_id)
        if region_year != cls._get_region_year(to_region_id):
            raise ValueError(
                f"Region years do not match: "
                f"{from_region_id}, {to_region_id}"
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
                f"No regions found in range: "
                f"{from_region_id}...{to_region_id}"
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
        from datasets.region.RegionTypeUtils import RegionTypeUtils

        region_type = RegionTypeUtils.get_region_type(region_id)
        region_year = cls._get_region_year(region_id)
        raw_regions = cls._get_raw_region_data_list_for_region_type(
            region_type, region_year
        )
        center_region = next(
            (r for r in raw_regions if r["region_id"] == region_id), None
        )
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
