import os

from lanka_data.api.data.DataSource import DataSource
from lanka_data.datasets.dataset.custom.GIG2Dataset import GIG2Dataset
from utils_future import Log

log = Log("Census2012Dataset")


class Census2012Dataset(GIG2Dataset):

    @classmethod
    def metadata_file_path(cls) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "census2012.datasets.json",
        )

    def get_sources(self):
        return [
            DataSource(
                name="Census of Population and Housing 2012",
                url="https://www.statistics.gov.lk"
                + "/Resource/en/Population"
                + "/CPH_2011/CPH_2012_5Per_Rpt.pdf",
            )
        ]

    def get_region_group(self) -> str:
        return "regions"

    def get_year(self) -> str:
        return "2012"

    @classmethod
    def get_supported_whens(cls):
        return ["2012"]
