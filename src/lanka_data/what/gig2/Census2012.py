import logging
import os
from functools import cache

from lanka_data.what.What import What
from utils_future import WWW, JSONFile

log = logging.getLogger(__name__)


class Census2012(What):
    WHAT_LABEL_TO_ID_FILE_PATH = os.path.join(
        "src", "lanka_data", "what", "gig2", "gig2.datasets.json"
    )

    def __init__(self, what_label: str, region_group: str, year: str):
        self.what_label = what_label
        self.region_group = region_group
        self.year = year

    @classmethod
    @cache
    def get_what_label_to_id(cls):
        return JSONFile(cls.WHAT_LABEL_TO_ID_FILE_PATH).read()

    @staticmethod
    def clean(d, region_idx):
        region_id = d["entity_id"]
        values = {}
        for k, v in d.items():
            if "total" in k:
                continue
            if "entity_id" in k:
                continue
            values[k] = int(float(v))

        values = dict(sorted(values.items(), key=lambda item: -item[1]))

        total_value = sum(values.values())
        p_values = {k: round(v / total_value, 2) for k, v in values.items()}
        return dict(
            region_id=region_id,
            region_name=region_idx[region_id]["name"],
            values=values,
            total_value=total_value,
            p_values=p_values,
        )

    @cache
    def get_results(self, regions) -> list[dict]:
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
            Census2012.clean(d, region_idx) for d in filtered_data_list
        ]
        return cleaned_data_list
