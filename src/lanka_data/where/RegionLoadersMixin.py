from lanka_data.where.RegionTypeUtils import RegionTypeUtils
from utils_future.GeoUtils import GeoUtils


class RegionLoadersMixin:
    @classmethod
    def from_token(cls, token: str):  # noqa: CFQ004, C901
        region_year = "Current"
        if ":" in token:
            parent_region_id, region_type = token.split(":")
            if "-pre" in parent_region_id:
                parent_region_id, region_year = parent_region_id.split("-pre")
            return cls.from_parent_region_id_and_region_type(
                region_type, parent_region_id, region_year
            )

        if "..." in token:
            from_region_id, to_region_id = token.split("...")
            return cls.from_region_id_range(
                from_region_id, to_region_id, region_year
            )

        if "@" in token:
            region_id, radius_km = token.split("@")
            return cls.from_region_radius(region_id, radius_km, region_year)

        if "&" in token:
            region_a_id, region_b_id = token.split("&")
            return cls.from_region_intersection(
                region_a_id, region_b_id, region_year
            )

        return cls.from_region_ids_str(token, region_year)

    @classmethod
    def from_region_intersection(cls, region_a_id, region_b_id, region_year):
        region_a_type = RegionTypeUtils.get_region_type(region_a_id)
        region_b_type = RegionTypeUtils.get_region_type(region_b_id)

        region_a_id_key = f"{region_a_type}_id"
        region_b_id_key = f"{region_b_type}_id"

        raw_gnds = cls._get_raw_region_data_list_for_region_type(
            "gnd", region_year
        )
        intersection_gnds = []
        for gnd in raw_gnds:
            if (
                gnd.get(region_a_id_key) == region_a_id
                and gnd.get(region_b_id_key) == region_b_id
            ):
                intersection_gnds.append(gnd)

        description = (
            "Intersection of "
            + f"{region_a_type.title()} {region_a_id}"
            + f" and {region_b_type.title()} {region_b_id}"
        ) + (f" (pre-{region_year} Map)" if region_year != "Current" else "")
        return cls(intersection_gnds, region_year, description)

    @staticmethod
    def _is_within_radius(radius_km, center_region, region):
        center_lat = center_region["center_lat"]
        center_lng = center_region["center_lng"]

        lat = region["center_lat"]
        lng = region["center_lng"]
        distance_km = GeoUtils.haversine_distance(
            center_lat, center_lng, lat, lng
        )
        return distance_km <= float(radius_km)

    @classmethod
    def from_region_radius(cls, region_id, radius_km, region_year):
        region_type = RegionTypeUtils.get_region_type(region_id)
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

        nearby_regions = [
            r
            for r in raw_regions
            if cls._is_within_radius(radius_km, center_region, r)
        ]
        if not nearby_regions:
            raise ValueError(
                f"No regions found within {radius_km} km of {region_id}"
            )

        description = (
            f"{region_type.title()}s"
            + f" within {radius_km} km of {region_id}"
            + (f" (pre-{region_year} Map)" if region_year != "Current" else "")
        )
        return cls(nearby_regions, region_year, description)

    @classmethod
    def from_region_id_range(cls, from_region_id, to_region_id, region_year):
        region_type = RegionTypeUtils.get_region_type(from_region_id)
        if region_type != RegionTypeUtils.get_region_type(to_region_id):
            raise ValueError(
                f"Region types do not match: {from_region_id}, {to_region_id}"
            )

        raw_regions = cls._get_raw_region_data_list_for_region_type(
            region_type, region_year
        )
        raw_regions = [
            d
            for d in raw_regions
            if from_region_id <= d["region_id"] <= to_region_id
        ]
        if not raw_regions:
            raise ValueError(
                f"No regions found in range: {from_region_id}...{to_region_id}"
            )
        description = (
            f"{region_type.title()}s,"
            + f" from {from_region_id} to {to_region_id}"
            + (f" (pre-{region_year} Map)" if region_year != "Current" else "")
        )
        return cls(raw_regions, region_year, description)

    @classmethod
    def from_region_ids_str(cls, region_ids_str, region_year):
        region_ids = region_ids_str.split(",")
        region_types = [
            RegionTypeUtils.get_region_type(region_id)
            for region_id in region_ids
        ]
        if len(set(region_types)) != 1:
            raise ValueError(
                f"All region IDs must be of the same type: {region_ids_str}"
            )

        region_type = region_types[0]
        raw_regions = cls._get_raw_region_data_list_for_region_type(
            region_type, region_year
        )
        raw_regions = [d for d in raw_regions if d["region_id"] in region_ids]
        if not raw_regions:
            raise ValueError(f"Region ID not found: {region_ids_str}")
        description = (
            f"{region_type.title()}s"
            + f" with IDs {region_ids_str}"
            + (f" (pre-{region_year} Map)" if region_year != "Current" else "")
        )
        return cls(raw_regions, region_year, description)

    @classmethod
    def is_parent(cls, region, parent_region_id) -> bool:  # noqa: CFQ004
        if parent_region_id == "LK":
            return True

        region_id = region["region_id"]
        if parent_region_id in region_id:
            return True

        parent_region_type = RegionTypeUtils.get_region_type(parent_region_id)
        parent_region_id_key = f"{parent_region_type}_id"
        if region.get(parent_region_id_key) == parent_region_id:
            return True

        return False

    @classmethod
    def from_parent_region_id_and_region_type(
        cls,
        region_type,
        parent_region_id,
        region_year,
    ):
        raw_regions = cls._get_raw_region_data_list_for_region_type(
            region_type, region_year
        )
        raw_regions = [
            region
            for region in raw_regions
            if cls.is_parent(region, parent_region_id)
        ]
        if not raw_regions:
            raise ValueError(
                f"No regions found for parent ID: {parent_region_id}"
            )
        description = (
            f"{region_type.title()}s"
            + f" within {parent_region_id}"
            + (f" (pre-{region_year} Map)" if region_year != "Current" else "")
        )
        return cls(raw_regions, region_year, description)
