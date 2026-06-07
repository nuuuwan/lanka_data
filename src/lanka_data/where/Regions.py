from functools import cached_property

from lanka_data.where.RegionLoadersMixin import RegionLoadersMixin
from lanka_data.where.RegionTypeUtils import RegionTypeUtils
from lanka_data.where.Where import Where
from utils_future import WWW, Log

log = Log("Regions")


class Regions(Where, RegionLoadersMixin):

    def __init__(self, raw_region_data_list: list[str], year: str):
        super().__init__(self.build_title(raw_region_data_list), year)
        self.raw_region_data_list = raw_region_data_list

    @classmethod
    def build_title(cls, raw_region_data_list):
        region_type = RegionTypeUtils.get_region_type(
            raw_region_data_list[0]["id"]
        )
        region_names = [d["name"] for d in raw_region_data_list]
        n_regions = len(region_names)
        if n_regions == 1:
            return region_names[0] + f" {region_type.title()}"
        if n_regions <= 5:
            return ", ".join(region_names) + f" {region_type.title()}s"

        return f"{n_regions} {region_type.title()}s"

    @cached_property
    def region_type(self):
        return RegionTypeUtils.get_region_type(
            self.raw_region_data_list[0]["id"]
        )

    @cached_property
    def region_to_current_ids(self):
        region_to_current_ids = {}
        for d in self.raw_region_data_list:
            region_id = d["id"]
            current_ids = d.get("current_ids")
            if current_ids is None:
                current_ids = [region_id]
            region_to_current_ids[region_id] = current_ids
        return region_to_current_ids

    @classmethod
    def _get_raw_region_data_list_for_region_type(
        cls, region_type: str, historical_year: str
    ):
        if historical_year is None:
            raise ValueError("historical_year cannot be None")
        if historical_year == "Current":
            url = (
                "https://raw.githubusercontent.com"
                + "/nuuuwan/lk_admin_regions/refs/heads/main"
                + "/data/ents"
                + f"/{region_type}s.json"
            )
        else:
            url = (
                "https://raw.githubusercontent.com"
                + "/nuuuwan/lk_admin_regions/refs/heads/main"
                + "/data/ents/history"
                + f"/{region_type}s-pre{historical_year}.json"
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
