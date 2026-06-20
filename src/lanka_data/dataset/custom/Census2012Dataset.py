import os

from lanka_data.data.FieldNameUtils import FieldNameUtils
from lanka_data.dataset.custom.GIG2Dataset import GIG2Dataset
from utils_future import Log

log = Log("Census2012Dataset")


class Census2012Dataset(GIG2Dataset):

    @classmethod
    def metadata_file_path(cls) -> str:
        return os.path.join(
            "src",
            "lanka_data",
            "dataset",
            "custom",
            "census2012.datasets.json",
        )

    def get_source_info_list(self) -> list[dict]:
        return [
            dict(
                label="Census of Population and Housing 2012",
                url="https://www.statistics.gov.lk"
                + "/Resource/en/Population"
                + "/CPH_2011/CPH_2012_5Per_Rpt.pdf",
            )
        ]

    def get_region_group(self) -> str:
        return "regions"

    def get_year(self) -> str:
        return "2012"

    def clean_data_row(self, row: dict) -> dict:
        d = {"region_id": row["entity_id"]}
        values = {}
        for k, v in row.items():
            if k in ["entity_id"]:
                continue
            values[FieldNameUtils.normalize(k)] = int(float(v))

        d["values"] = values
        return d
