from abc import abstractmethod

from lanka_data.data.DataSource import DataSource
from lanka_data.data.FieldNameUtils import FieldNameUtils
from lanka_data.dataset.RegionValueDataset import RegionValueDataset
from utils_future import WWW, JSONFile, Log

log = Log("GIG2Dataset")


class GIG2Dataset(RegionValueDataset):
    def __init__(self, region_data_list: list[dict], table_id: str):
        RegionValueDataset.__init__(self, region_data_list)
        self.table_id = table_id

    @classmethod
    def from_label_and_region_data_list(
        cls, label: str, region_data_list: list[dict]
    ):
        label_to_table_id = cls.get_label_to_table_id()
        if label not in label_to_table_id:
            raise ValueError(f"Label '{label}' not found in metadata.")
        table_id = label_to_table_id[label]
        return cls(region_data_list, table_id)

    @classmethod
    @abstractmethod
    def metadata_file_path(cls) -> str:
        pass

    @classmethod
    def get_label_to_table_id(cls) -> dict[str, str]:
        return JSONFile(cls.metadata_file_path()).read()

    @classmethod
    def get_labels(cls) -> list[str]:
        return list(cls.get_label_to_table_id().keys())

    @abstractmethod
    def get_sources(self) -> list[DataSource]:
        pass

    @abstractmethod
    def get_region_group(self) -> str:
        pass

    @abstractmethod
    def get_year(self) -> str:
        pass

    def get_source_data_table(self) -> list[dict]:
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/gig-data/refs/heads/master"
            + "/gig2"
            + f"/{self.table_id}.{self.get_region_group()}.{self.get_year()}"
            + ".tsv"
        )
        return WWW(url).read_tsv()

    def clean_data_row(self, row: dict) -> dict:
        d = {"region_id": row["entity_id"]}
        values = {}
        for k, v in row.items():
            if k in ["entity_id"]:
                continue
            if "total" in k:
                continue
            values[FieldNameUtils.normalize(k)] = int(float(v))

        d["values"] = values
        return d
