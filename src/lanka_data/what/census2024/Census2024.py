import os
from functools import cache

from lanka_data.what.What import What
from utils_future import WWW, JSONFile, Log

log = Log("Census2024")


class Census2024(What):

    def __init__(self, title: str):
        super().__init__(title)

    @classmethod
    def metadata_file_path(cls) -> str:
        return os.path.join(
            "src",
            "lanka_data",
            "what",
            "census2024",
            "census2024.metadata.json",
        )

    @classmethod
    def get_metadata(cls) -> dict:
        return JSONFile(cls.metadata_file_path()).read()

    @classmethod
    @cache
    def get_title_to_path(cls) -> dict:
        metadata = cls.get_metadata()
        label_to_path = {}
        for level1 in metadata:
            for level2, label in metadata[level1].items():
                if label in label_to_path:
                    continue
                path = f"{level1}/{level2}"
                label_to_path[label] = path
        return label_to_path

    @classmethod
    def clean(cls, d):
        values = {}
        for k, v in d.items():
            if "region" in k:
                continue
            if "total" in k:
                continue
            values[k] = int(float(v))
        values = dict(sorted(values.items(), key=lambda item: -item[1]))
        total_value = sum(values.values())
        pct_values = {k: round(v / total_value, 4) for k, v in values.items()}

        return dict(
            values=values,
            total_value=total_value,
            pct_values=pct_values,
        )

    @classmethod
    def get_source_info(cls):
        return dict(
            source="Census of Population and Housing 2024",
            source_url="https://www.statistics.gov.lk"
            + "/Population/StaticalInformation/CPH2024",
        )

    def get_source_data_list(self) -> list[dict]:
        label_to_path = self.get_title_to_path()
        path = label_to_path.get(self.title)
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_census_2024/refs/heads/main"
            + f"/data/{path}/data.json"
        )
        raw_data_list = WWW(url).read_json()
        return raw_data_list
