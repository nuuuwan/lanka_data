import os
from functools import cache

from lanka_data.api.what.FieldNameUtils import FieldNameUtils
from lanka_data.api.what.What import What
from utils_future import WWW, JSONFile, Log

log = Log("Census2024")


class Census2024(What):

    @classmethod
    def metadata_file_path(cls) -> str:
        return os.path.join(
            "src",
            "lanka_data",
            "api",
            "what",
            "census2024",
            "census2024.metadata.json",
        )

    @classmethod
    def get_metadata(cls) -> dict:
        return JSONFile(cls.metadata_file_path()).read()

    @classmethod
    @cache
    def get_label_to_data_table_id(cls) -> dict:
        metadata = cls.get_metadata()
        return {entry["label"]: entry["data_table_id"] for entry in metadata}

    @classmethod
    @cache
    def get_label_to_description(cls) -> dict:
        metadata = cls.get_metadata()
        return {entry["label"]: entry["description"] for entry in metadata}

    def get_description(self):
        return self.get_label_to_description().get(self.title, "")

    @classmethod
    def get_what_to_whens(cls) -> dict[str, set[str]]:
        label_to_data_table_id = cls.get_label_to_data_table_id()
        what_to_whens = {}
        for label in label_to_data_table_id:
            what_to_whens[label] = ["2024"]
        return what_to_whens

    @classmethod
    def extract_source_data_values(cls, d):
        values = {
            FieldNameUtils.normalize(k): int(v)
            for k, v in d["values"].items()
        }

        return dict(values=values)

    @classmethod
    def clean(cls, d):
        values = {k: v for k, v in d.items()}
        total_value = d["total_value"]
        pct_values = {k: round(v / total_value, 4) for k, v in values.items()}

        return dict(
            values=values,
            total_value=total_value,
            pct_values=pct_values,
        )

    def get_source_info_list(self):
        description = self.get_label_to_description().get(self.title, "")
        return [
            dict(
                label="Census of Population and Housing 2024",
                url="https://www.statistics.gov.lk"
                + "/Population/StaticalInformation/CPH2024",
                description=description,
            )
        ]

    def get_source_data_list(self) -> list[dict]:
        label_to_data_table_id = self.get_label_to_data_table_id()
        data_table_id = label_to_data_table_id.get(self.title)
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_census_2024/refs/heads/main"
            + f"/data/{data_table_id}/data.json"
        )
        raw_data_list = WWW(url).read_json()

        return raw_data_list
