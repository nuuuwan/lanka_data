import os
from abc import abstractmethod

from api.data.DataSource import DataSource
from datasets.data.FieldNameUtils import FieldNameUtils
from api.dataset.RegionValueDataset import RegionValueDataset
from api.utils_future import WWW, JSONFile, Log

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

    def fix_lg_id_bug(self, d_list: dict) -> dict:
        corrections_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "lg.corrections.json"
        )
        corrections_file = JSONFile(corrections_path)
        old_id_to_new_id = corrections_file.read()
        new_d_list = []
        for d in d_list:
            old_id = d["entity_id"]
            new_id = old_id_to_new_id.get(old_id, old_id)
            d["entity_id"] = new_id
            new_d_list.append(d)
        return new_d_list

    def get_source_data_table(self) -> list[dict]:
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/gig-data/refs/heads/master"
            + "/gig2"
            + f"/{self.table_id}.{self.get_region_group()}.{self.get_year()}"
            + ".tsv"
        )

        d_list = WWW(url).read_tsv()
        d_list = self.fix_lg_id_bug(d_list)
        return d_list

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
