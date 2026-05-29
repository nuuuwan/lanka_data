from functools import cached_property

from lanka_data.where.RegionTypeUtils import RegionTypeUtils
from lanka_data.where.Where import Where
from utils_future import WWW, Log
from utils_future.GeoUtils import GeoUtils

log = Log("Regions")


class Regions(Where):

    def __init__(self, regions: list[str]):
        self.regions = regions

    @cached_property
    def region_type(self):
        return RegionTypeUtils.get_region_type(self.regions[0]["id"])

    @classmethod
    def _get_data_list_for_region_type(cls, region_type: str):

        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main"
            + f"/data/ents/{region_type}s.json"
        )
        return WWW(url).read_json()

    @classmethod
    def from_token(cls, token: str):  # noqa: CFQ004
        if ":" in token:
            parent_region_id, region_type = token.split(":")
            return Regions.from_parent_region_id_and_region_type(
                region_type, parent_region_id
            )

        if "..." in token:
            from_region_id, to_region_id = token.split("...")
            return Regions.from_region_id_range(from_region_id, to_region_id)

        if "@" in token:
            region_id, radius_km = token.split("@")
            return Regions.from_region_radius(region_id, radius_km)

        if "&" in token:
            region_a_id, region_b_id = token.split("&")
            return Regions.from_region_intersection(region_a_id, region_b_id)

        return Regions.from_region_ids_str(token)

    @classmethod
    def from_region_intersection(cls, region_a_id, region_b_id):
        region_a_type = RegionTypeUtils.get_region_type(region_a_id)
        region_b_type = RegionTypeUtils.get_region_type(region_b_id)

        region_a_id_key = f"{region_a_type}_id"
        region_b_id_key = f"{region_b_type}_id"

        gnds = cls._get_data_list_for_region_type("gnd")
        intersection_gnds = []
        for gnd in gnds:
            if (
                gnd.get(region_a_id_key) == region_a_id
                and gnd.get(region_b_id_key) == region_b_id
            ):
                intersection_gnds.append(gnd)

        return cls(intersection_gnds)

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
    def from_region_radius(cls, region_id, radius_km):
        region_type = RegionTypeUtils.get_region_type(region_id)
        regions = cls._get_data_list_for_region_type(region_type)

        center_region = None
        for region in regions:
            if region["id"] == region_id:
                center_region = region
                break

        if not center_region:
            raise ValueError(f"Region ID not found: {region_id}")

        nearby_regions = [
            r
            for r in regions
            if cls._is_within_radius(radius_km, center_region, r)
        ]
        if not nearby_regions:
            raise ValueError(
                f"No regions found within {radius_km} km of {region_id}"
            )

        return cls(nearby_regions)

    @classmethod
    def from_region_id_range(cls, from_region_id, to_region_id):
        region_type = RegionTypeUtils.get_region_type(from_region_id)
        if region_type != RegionTypeUtils.get_region_type(to_region_id):
            raise ValueError(
                f"Region types do not match: {from_region_id}, {to_region_id}"
            )

        regions = cls._get_data_list_for_region_type(region_type)
        regions = [
            d for d in regions if from_region_id <= d["id"] <= to_region_id
        ]
        if not regions:
            raise ValueError(
                f"No regions found in range: {from_region_id}...{to_region_id}"
            )
        return cls(regions)

    @classmethod
    def from_region_ids_str(cls, region_ids_str):
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
        regions = cls._get_data_list_for_region_type(region_type)
        regions = [d for d in regions if d["id"] in region_ids]
        if not regions:
            raise ValueError(f"Region ID not found: {region_ids_str}")
        return cls(regions)

    @classmethod
    def is_parent(cls, region, parent_region_id) -> bool:  # noqa: CFQ004
        if parent_region_id == "LK":
            return True

        region_id = region["id"]
        if parent_region_id in region_id:
            return True

        parent_region_type = RegionTypeUtils.get_region_type(parent_region_id)
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

    @classmethod
    def clean(cls, d):
        new_d = {
            "region_id": d["id"],
            "region_name": d["name"],
            "region_type": RegionTypeUtils.get_region_type(d["id"]),
        }
        for k, v in d.items():
            if k in ["id", "name"]:
                continue
            new_d[k] = v

        return new_d

    def get_result(self) -> list[dict]:
        data_list = [self.clean(d) for d in self.regions]
        return dict(
            data_list=data_list,
            source="Department of Census and Statistics, Sri Lanka",
            source_url="https://www.statistics.gov.lk/",
        )
