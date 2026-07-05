from functools import cached_property

from datasets.region.RegionLoadersMixin import RegionLoadersMixin
from datasets.region.RegionTypeUtils import RegionTypeUtils
from datasets.region.Where import Where
from api.utils_future import Log

log = Log("Regions")


class Regions(Where, RegionLoadersMixin):

    def __init__(
        self,
        raw_region_data_list: list[str],
        region_year: str,
    ):
        super().__init__(
            self.build_title(raw_region_data_list),
            region_year,
        )
        self.raw_region_data_list = raw_region_data_list

    @classmethod
    def build_title(cls, raw_region_data_list):
        region_type = RegionTypeUtils.get_region_type(
            raw_region_data_list[0]["region_id"]
        )
        region_names = [d["region_name"] for d in raw_region_data_list]
        n_regions = len(region_names)
        if n_regions == 1:
            return region_names[0] + f" {region_type.title()}"
        if n_regions <= 5:
            return ", ".join(region_names) + f" {region_type.title()}s"

        return f"{n_regions} {region_type.title()}s"

    @cached_property
    def region_ids(self):
        return [d["region_id"] for d in self.raw_region_data_list]

    @cached_property
    def region_type(self):
        return RegionTypeUtils.get_region_type(
            self.raw_region_data_list[0]["region_id"]
        )

    @cached_property
    def region_to_current_ids(self):
        region_to_current_ids = {}
        for d in self.raw_region_data_list:
            region_id = d["region_id"]
            current_ids = d.get("current_ids")
            if current_ids is None:
                current_ids = [region_id]
            region_to_current_ids[region_id] = current_ids
        return region_to_current_ids

    @classmethod
    def clean(cls, d):
        new_d = {
            "region_id": d["region_id"],
            "region_name": d["region_name"],
            "region_type": RegionTypeUtils.get_region_type(d["region_id"]),
        }
        for k, v in d.items():
            if k in ["region_id", "region_name"]:
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
