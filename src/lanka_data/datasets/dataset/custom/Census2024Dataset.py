import os

from lanka_data.datasets.data import FieldNameUtils
from lanka_data.api.data.DataSource import DataSource
from lanka_data.api.dataset.RegionValueDataset import RegionValueDataset
from utils_future import WWW, JSONFile, Log

log = Log("Census2024Dataset")


class Census2024Dataset(RegionValueDataset):
    def __init__(self, region_data_list: list[dict], table_id: str):
        RegionValueDataset.__init__(self, region_data_list)
        self.table_id = table_id

    def get_year(self):
        return "2024"

    @classmethod
    def get_supported_whens(cls):
        return ["2024"]

    @classmethod
    def from_label_and_region_data_list(
        cls, label: str, region_data_list: list[dict]
    ) -> "Census2024Dataset":
        label_to_table_id = cls.get_label_to_table_id()
        if label not in label_to_table_id:
            raise ValueError(f"Label '{label}' not found in metadata.")
        table_id = label_to_table_id[label]
        return cls(region_data_list, table_id)

    @classmethod
    def metadata_file_path(cls) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "census2024.metadata.json",
        )

    @classmethod
    def get_label_to_table_id(cls) -> dict[str, str]:
        metadata = JSONFile(cls.metadata_file_path()).read()
        return {entry["label"]: entry["table_id"] for entry in metadata}

    @classmethod
    def get_labels(cls) -> list[str]:
        return list(cls.get_label_to_table_id().keys())

    def get_sources(self):
        return [
            DataSource(
                name="Census of Population and Housing 2024",
                url="https://www.statistics.gov.lk"
                + "/Population/StaticalInformation/CPH2024",
            )
        ]

    def get_source_data_table(self) -> list[dict]:
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_census_2024/refs/heads/main"
            + f"/data/{self.table_id}/data.json"
        )
        return WWW(url).read_json()

    def clean_data_row(self, row: dict) -> dict:
        row["values"] = {
            FieldNameUtils.normalize(k): int(float(v))
            for k, v in row["values"].items()
        }
        return row
