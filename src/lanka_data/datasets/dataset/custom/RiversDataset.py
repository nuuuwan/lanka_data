from lanka_data.api.data.DataSource import DataSource
from lanka_data.api.dataset.RegionValueDataset.RegionValueDataset import RegionValueDataset
from lanka_data.datasets.region.rivers.RiversData import (
    LABEL_CATCHMENT,
    LABEL_RIVER_LEN,
    RiversData,
)

RIVERS_WHEN = "2026"


class RiversDataset(RegionValueDataset):
    def __init__(self, region_data_list: list[dict], label: str):
        RegionValueDataset.__init__(self, region_data_list)
        self.label = label

    def get_year(self):
        return RIVERS_WHEN

    @classmethod
    def get_supported_whens(cls):
        return [RIVERS_WHEN]

    @classmethod
    def get_labels(cls) -> list[str]:
        return [LABEL_RIVER_LEN, LABEL_CATCHMENT]

    @classmethod
    def supports(cls, label: str, when: str) -> bool:
        return when in cls.get_supported_whens() and label in cls.get_labels()

    @classmethod
    def from_label_and_region_data_list(
        cls, label: str, region_data_list: list[dict]
    ) -> "RiversDataset":
        return cls(region_data_list, label)

    def get_sources(self):
        return [
            DataSource(
                name="HydroRIVERS (via lk_rivers)",
                url="https://github.com/nuuuwan/lk_rivers",
            )
        ]

    def get_source_data_table(self) -> list[dict]:
        measures = RiversData.get_river_measures()
        return [
            {"region_id": region_id, "values": {self.label: row[self.label]}}
            for region_id, row in measures.items()
        ]

    def clean_data_row(self, row: dict) -> dict:
        return row
