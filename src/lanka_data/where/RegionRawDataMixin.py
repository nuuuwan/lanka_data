from functools import cache

from lanka_data.where.RegionTypeUtils import RegionTypeUtils
from utils_future import WWW


class RegionRawDataMixin:
    @classmethod
    def _get_url(cls, region_type, region_year):
        if region_year == "Current":
            return (
                "https://raw.githubusercontent.com"
                + "/nuuuwan/lk_admin_regions/refs/heads/main"
                + "/data/ents"
                + f"/{region_type}s.json"
            )

        return (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main"
            + "/data/ents/history"
            + f"/{region_type}s-pre{region_year}.json"
        )

    @classmethod
    @cache
    def _get_raw_region_data_list_for_region_type(
        cls, region_type: str, region_year: str
    ):
        if region_year is None:
            raise ValueError("region_year cannot be None")
        url = cls._get_url(region_type, region_year)
        raw_data_list = WWW(url).read_json()

        def remap(d):
            d = (
                dict(
                    region_id=d["id"],
                    region_name=d["name"],
                    region_type=region_type,
                    history_year=region_year,
                )
                | d
            )
            del d["id"]
            del d["name"]
            return d

        remapped_data_list = [remap(d) for d in raw_data_list]
        return remapped_data_list

    @classmethod
    def _get_raw_region_data_idx_for_region_type(
        cls, region_type: str, region_year: str
    ):
        raw_region_data_list = cls._get_raw_region_data_list_for_region_type(
            region_type, region_year
        )
        return {d["region_id"]: d for d in raw_region_data_list}

    @classmethod
    def _get_region_year(cls, region_id):
        if "-pre" in region_id:
            return region_id.split("-pre")[1]
        return "Current"

    @classmethod
    def _get_raw_region_data_for_region_id(cls, region_id: str):
        return cls._get_raw_region_data_list_for_region_ids([region_id])[0]

    @classmethod
    def _get_raw_region_data_list_for_region_ids(cls, region_ids: list[str]):
        idx = {}  # region_year -> region_type -> region_ids
        for region_id in region_ids:
            region_year = cls._get_region_year(region_id)
            region_type = RegionTypeUtils.get_region_type(region_id)
            if region_year not in idx:
                idx[region_year] = {}
            if region_type not in idx[region_year]:
                idx[region_year][region_type] = set()
            idx[region_year][region_type].add(region_id)

        region_id_to_raw_region = {}
        for region_year, region_type_to_ids in idx.items():
            for region_type, region_ids in region_type_to_ids.items():
                raw_region_idx = cls._get_raw_region_data_idx_for_region_type(
                    region_type, region_year
                )
                for region_id in region_ids:
                    raw_region = raw_region_idx.get(region_id)
                    if raw_region is None:
                        raise ValueError(
                            f"Region ID not found: {region_id} (type: {region_type}, year: {region_year})"
                        )
                    region_id_to_raw_region[region_id] = raw_region

        return [
            region_id_to_raw_region[region_id] for region_id in region_ids
        ]

    # flake8: noqa: C901
    @classmethod
    def is_parent(cls, parent_region_id: str, child_region_id: str) -> bool:
        parent_region_id = parent_region_id.split("-pre")[0]
        if parent_region_id == "LK":
            return True

        if parent_region_id in child_region_id:
            return True

        parent_region_type = RegionTypeUtils.get_region_type(parent_region_id)
        child_raw_region = cls._get_raw_region_data_for_region_id(
            child_region_id
        )

        parent_id_key = f"{parent_region_type}_id"
        if (
            parent_id_key in child_raw_region
            and child_raw_region[parent_id_key] == parent_region_id
        ):
            return True

        return False

    @classmethod
    def has_some_parent(cls, region_id, parent_region_ids):
        for parent_region_id in parent_region_ids:
            if cls.is_parent(parent_region_id, region_id):
                return True
        return False
