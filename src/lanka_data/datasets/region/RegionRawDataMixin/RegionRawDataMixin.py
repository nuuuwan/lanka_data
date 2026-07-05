from functools import cache

from lanka_data.datasets.region.RegionRawDataMixin.RegionFetchMixin import \
    RegionFetchMixin
from lanka_data.datasets.region.RegionRawDataMixin.RegionParentMixin import \
    RegionParentMixin
from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils


class RegionRawDataMixin(RegionParentMixin, RegionFetchMixin):
    @classmethod
    @cache
    def _get_url(cls, region_type, region_year):
        base = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main/data/ents"
        )
        if region_year == "Current":
            return f"{base}/{region_type}s.json"
        return f"{base}/history/{region_type}s-pre{region_year}.json"

    @classmethod
    def _populate_from_type(
        cls, region_id_to_raw_region, r_ids, region_type, region_year
    ):
        raw_idx = cls._get_raw_region_data_idx_for_region_type(
            region_type, region_year
        )
        for region_id in r_ids:
            raw_region = raw_idx.get(region_id)
            if raw_region is None:
                raise ValueError(
                    f"Region ID not found: {region_id} "
                    f"(type: {region_type}, year: {region_year})"
                )
            region_id_to_raw_region[region_id] = raw_region

    @classmethod
    def _get_raw_region_data_list_for_region_ids(cls, region_ids: list[str]):
        idx = {}
        for region_id in region_ids:
            region_year = cls._get_region_year(region_id)
            region_type = RegionTypeUtils.get_region_type(region_id)
            idx.setdefault(region_year, {}).setdefault(
                region_type, set()
            ).add(region_id)
        region_id_to_raw_region = {}
        for region_year, region_type_to_ids in idx.items():
            for region_type, r_ids in region_type_to_ids.items():
                cls._populate_from_type(
                    region_id_to_raw_region, r_ids, region_type, region_year
                )
        return [
            region_id_to_raw_region[region_id] for region_id in region_ids
        ]
