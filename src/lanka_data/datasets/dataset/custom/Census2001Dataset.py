import os

from lanka_data.api.data.DataSource import DataSource
from lanka_data.api.dataset.RegionValueDataset.RegionValueDataset import RegionValueDataset
from lanka_data.datasets.data.FieldNameUtils import FieldNameUtils
from utils_future import WWW, JSONFile, Log

log = Log("Census2001Dataset")


class Census2001Dataset(RegionValueDataset):
    def __init__(self, region_data_list: list[dict], table_id: str):
        RegionValueDataset.__init__(self, region_data_list)
        self.table_id = table_id

    def get_year(self):
        return "2001"

    @classmethod
    def get_supported_whens(cls):
        return ["2001"]

    @classmethod
    def from_label_and_region_data_list(
        cls, label: str, region_data_list: list[dict]
    ) -> "Census2001Dataset":
        label_to_table_id = cls.get_label_to_table_id()
        if label not in label_to_table_id:
            raise ValueError(f"Label '{label}' not found in metadata.")
        table_id = label_to_table_id[label]
        return cls(region_data_list, table_id)

    @classmethod
    def metadata_file_path(cls) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "census2001.metadata.json",
        )

    @classmethod
    def get_label_to_table_id(cls) -> dict[str, str]:
        return JSONFile(cls.metadata_file_path()).read()

    @classmethod
    def get_labels(cls) -> list[str]:
        return list(cls.get_label_to_table_id().keys())

    def get_sources(self):
        return [
            DataSource(
                name="Census of Population and Housing 2001",
                url="https://www.statistics.gov.lk"
                + "/Population/StaticalInformation/CPH2001",
            )
        ]

    def get_source_data_table(self) -> list[dict]:
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_census_2001/refs/heads/main"
            + f"/data/{self.table_id}/normalized_data.json"
        )
        return WWW(url).read_json()

    def clean_data_row(self, row: dict) -> dict:
        row["values"] = {
            FieldNameUtils.normalize(k): int(float(v))
            for k, v in row["values"].items()
        }
        return row
