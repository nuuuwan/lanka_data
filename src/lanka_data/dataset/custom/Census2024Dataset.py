import os

from lanka_data.dataset.Dataset import Dataset
from utils_future import WWW, JSONFile, Log

log = Log("Census2024")


class Census2024Dataset(Dataset):
    def __init__(self, table_id: str):
        super().__init__()
        self.table_id = table_id

    @classmethod
    def from_label(cls, label: str) -> "Census2024Dataset":
        label_to_table_id = cls.get_label_to_table_id()
        if label not in label_to_table_id:
            raise ValueError(f"Label '{label}' not found in metadata.")
        table_id = label_to_table_id[label]
        return cls(table_id)

    @classmethod
    def metadata_file_path(cls) -> str:
        return os.path.join(
            "src",
            "lanka_data",
            "dataset",
            "custom",
            "census2024.metadata.json",
        )

    @classmethod
    def get_label_to_table_id(cls) -> dict[str, str]:
        metadata = JSONFile(cls.metadata_file_path()).read()
        return {entry["label"]: entry["table_id"] for entry in metadata}

    @classmethod
    def get_labels(cls) -> list[str]:
        return list(cls.get_label_to_table_id().keys())

    @classmethod
    def get_source_info_list(cls) -> list[dict]:
        return [
            dict(
                label="Census of Population and Housing 2024",
                url="https://www.statistics.gov.lk"
                + "/Population/StaticalInformation/CPH2024",
            )
        ]

    def get_source_data(self) -> list[dict]:
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_census_2024/refs/heads/main"
            + f"/data/{self.table_id}/data.json"
        )
        return WWW(url).read_json()
