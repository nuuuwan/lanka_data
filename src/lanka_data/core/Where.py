from functools import cache, cached_property

import requests


class Where:
    def __init__(self, region_id: str):
        self.region_id = region_id

    def run(self):
        return self.region

    @cached_property
    def region_type(self) -> str:
        id_len = len(self.region_id)
        if self.region_id.startswith("LK"):
            return {
                2: "country",
                4: "province",
                5: "district",
                7: "dsd",
                10: "gnd",
            }.get(id_len, None)

        if self.region_id.startswith("EC-"):
            return {
                5: "ed",
                6: "pd",
            }.get(id_len, None)

        return None

    @classmethod
    @cache
    def get_data_idx_for_region_type(cls, region_type: str):

        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_admin_regions/refs/heads/main/data/ents"
            + f"/{region_type}s.json"
        )
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch data from {url}: {e}")

        data = response.json()
        data_idx = {d["id"]: d for d in data}
        return data_idx

    @cached_property
    def region(self):
        region_type = self.region_type
        if region_type is None:
            raise ValueError(f"Invalid region ID format: {self.region_id}")

        data_idx = self.get_data_idx_for_region_type(region_type)
        if self.region_id not in data_idx:
            raise ValueError(f"Region ID {self.region_id} not found in data")
        return dict(region_type=region_type) | data_idx[self.region_id]
