from functools import cached_property

from lanka_data.where.RegionLoadersMixin import RegionLoadersMixin
from lanka_data.where.RegionTypeUtils import RegionTypeUtils
from lanka_data.where.Where import Where
from utils_future import WWW, Log

log = Log("Regions")


class Regions(Where, RegionLoadersMixin):

    @classmethod
    def build_title(cls, raw_region_data_list):
        region_ids = [d["id"] for d in raw_region_data_list]
        n_regions = len(region_ids)
        if n_regions <= 10:
            return ", ".join(region_ids)

        return f"{n_regions} regions"

    def __init__(self, raw_region_data_list: list[str]):
        super().__init__(self.build_title(raw_region_data_list))
        self.raw_region_data_list = raw_region_data_list

    @cached_property
    def region_type(self):
        return RegionTypeUtils.get_region_type(
            self.raw_region_data_list[0]["id"]
        )

    @classmethod
    def _get_raw_region_data_list_for_region_type(cls, region_type: str):

        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main"
            + f"/data/ents/{region_type}s.json"
        )
        return WWW(url).read_json()

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
        data_list = [self.clean(d) for d in self.raw_region_data_list]
        return dict(
            data_list=data_list,
            source="Department of Census and Statistics, Sri Lanka",
            source_url="https://www.statistics.gov.lk/",
        )
