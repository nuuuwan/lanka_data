from functools import cache

from lanka_data.what.What import What
from utils_future import WWW, JSONFile


class GIG2(What):

    def __init__(self, what_label: str, region_group: str, year: str):
        self.what_label = what_label
        self.region_group = region_group
        self.year = year

    @classmethod
    @cache
    def get_what_label_to_id(cls):
        return JSONFile(cls.get_what_label_to_id_file_path()).read()

    @classmethod
    def clean(cls, d, region_idx):
        region_id = d["entity_id"]

        for k, v in d.items():
            if "total" in k:
                continue
            if "entity_id" in k:
                continue

        return dict(
            region_id=region_id,
            region_name=region_idx[region_id]["name"],
        ) | cls.get_custom_result(d)

    @cache
    def get_data_list(self, regions) -> list[dict]:
        what_label_to_id = self.get_what_label_to_id()
        what_id = what_label_to_id.get(self.what_label)
        if what_id is None:
            raise ValueError(f"Invalid what label: {self.what_label}")
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/gig-data/refs/heads/master"
            + f"/gig2/{what_id}.{self.region_group}.{self.year}.tsv"
        )
        data_list = WWW(url).read_tsv()

        region_idx = {r["id"]: r for r in regions.regions}
        filtered_data_list = [
            d for d in data_list if d["entity_id"] in region_idx
        ]
        cleaned_data_list = [
            self.clean(d, region_idx) for d in filtered_data_list
        ]
        return cleaned_data_list
