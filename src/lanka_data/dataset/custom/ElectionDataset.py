import os

from lanka_data.data.DataSource import DataSource
from lanka_data.dataset.custom.GIG2Dataset import GIG2Dataset
from utils_future import Log

log = Log("ElectionDataset")


class ElectionDataset(GIG2Dataset):

    def __init__(self, region_data_list: list[dict], table_id: str, year: str):
        GIG2Dataset.__init__(self, region_data_list, table_id)
        self.year = year

    @classmethod
    def from_label_and_region_data_list_and_year(
        cls, label: str, region_data_list: list[dict], year: str
    ) -> "ElectionDataset":
        table_id = cls.get_label_to_table_id()[label]
        return cls(region_data_list, table_id, year)

    def get_year(self) -> str:
        return self.year

    @classmethod
    def metadata_file_path(cls) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "elections.datasets.json",
        )

    def get_sources(self) -> list[dict]:
        return [
            DataSource(
                name="Election Commission of Sri lanka",
                url="https://www.elections.gov.lk",
            )
        ]

    def get_region_group(self) -> str:
        return "regions-ec"

    def clean_data_row(self, row: dict) -> dict:
        d = {"region_id": row["entity_id"]}
        values = {}
        for k, v in row.items():
            if k in ["entity_id", "electors", "polled", "valid", "rejected"]:
                continue
            if "total" in k:
                continue
            values[k] = int(float(v))

        d["values"] = values
        return d
