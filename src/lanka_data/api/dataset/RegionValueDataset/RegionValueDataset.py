from abc import abstractmethod
from functools import cached_property

from lanka_data.api.dataset.Dataset import Dataset
from lanka_data.api.dataset.RegionValueDataset.RegionValueDatasetTableMixin import (
    RegionValueDatasetTableMixin,
)


class RegionValueDataset(RegionValueDatasetTableMixin, Dataset):
    PCT_VALUE_PRECISION = 4

    def __init__(self, region_data_list):
        Dataset.__init__(self)
        self.region_data_list = region_data_list

    @abstractmethod
    def get_year(self):
        pass

    @cached_property
    def region_ids(self):
        return [d["region_id"] for d in self.region_data_list]

    @cached_property
    def region_id_to_current_ids(self):
        return {
            d["region_id"]: d.get("current_ids", [d["region_id"]])
            for d in self.region_data_list
        }

    @cached_property
    def region_idx(self):
        return {d["region_id"]: d for d in self.region_data_list}

    def get_region(self, region_id):
        return self.region_idx[region_id]

    def expand_and_clean(self, data):
        region_id = data["region_id"]
        region = self.get_region(region_id)
        expanded_data = dict(
            region_id=data["region_id"],
            region_name=region["region_name"],
            center_lat=region["center_lat"],
            center_lng=region["center_lng"],
            current_ids=region.get("current_ids", [region_id]),
        )
        values = dict(
            sorted(data["values"].items(), key=lambda item: -item[1])
        )
        expanded_data["values"] = values
        total_value = sum(values.values())
        expanded_data["total_value"] = total_value
        expanded_data["pct_values"] = {
            k: (
                round(v / total_value, self.PCT_VALUE_PRECISION)
                if total_value > 0
                else 0
            )
            for k, v in values.items()
        }
        return expanded_data

    @abstractmethod
    def get_source_data_table(self) -> list[dict]:
        pass

    @abstractmethod
    def clean_data_row(self, row: dict) -> dict:
        pass

    def get_complete_data_table(self) -> list[dict]:
        return [
            self.clean_data_row(row) for row in self.get_source_data_table()
        ]

    def has_values(self) -> bool:
        return True
