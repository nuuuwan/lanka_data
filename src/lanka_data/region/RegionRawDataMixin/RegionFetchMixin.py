from functools import cache

from utils_future import WWW


class RegionFetchMixin:
    @staticmethod
    def _remap_region_data(d, region_type, region_year):
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

    @classmethod
    @cache
    def _get_raw_region_data_list_for_region_type(
        cls, region_type: str, region_year: str
    ):
        if region_year is None:
            raise ValueError("region_year cannot be None")
        raw_data_list = WWW(
            cls._get_url(region_type, region_year)
        ).read_json()
        return [
            cls._remap_region_data(d, region_type, region_year)
            for d in raw_data_list
        ]

    @classmethod
    @cache
    def _get_raw_region_data_idx_for_region_type(
        cls, region_type: str, region_year: str
    ):
        return {
            d["region_id"]: d
            for d in cls._get_raw_region_data_list_for_region_type(
                region_type, region_year
            )
        }

    @classmethod
    @cache
    def _get_region_year(cls, region_id):
        if "-pre" in region_id:
            return region_id.split("-pre")[1]
        return "Current"

    @classmethod
    @cache
    def _get_raw_region_data_for_region_id(cls, region_id: str):
        return cls._get_raw_region_data_list_for_region_ids([region_id])[0]
