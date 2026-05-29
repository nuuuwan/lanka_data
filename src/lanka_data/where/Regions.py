from functools import cached_property

from lanka_data.where.RegionLoadersMixin import RegionLoadersMixin
from lanka_data.where.RegionTypeUtils import RegionTypeUtils
from lanka_data.where.Where import Where
from utils_future import WWW, Log

log = Log("Regions")


class Regions(Where, RegionLoadersMixin):

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
