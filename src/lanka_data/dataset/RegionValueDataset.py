from abc import abstractmethod
from functools import cached_property

from lanka_data.dataset.Dataset import Dataset


class RegionValueDataset(Dataset):
    PCT_VALUE_PRECISION = 4

    def __init__(self, region_data_list):
        Dataset.__init__(self)
        self.region_data_list = region_data_list

    @cached_property
    def region_ids(self):
        return [d["region_id"] for d in self.region_data_list]

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
        )
        values = {k: v for k, v in data["values"].items()}
        values = dict(sorted(values.items(), key=lambda item: -item[1]))
        expanded_data["values"] = values

        total_value = sum(values.values())
        expanded_data["total_value"] = total_value
        pct_values = {
            k: round(v / total_value, self.PCT_VALUE_PRECISION)
            for k, v in values.items()
        }
        expanded_data["pct_values"] = pct_values
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

    def get_data_table(self):
        complete_data_table = self.get_complete_data_table()
        filtered_data_table = [
            row
            for row in complete_data_table
            if row["region_id"] in self.region_ids
        ]
        sorted_data_table = sorted(
            filtered_data_table,
            key=lambda row: row["region_id"],
        )
        expanded_data_table = [
            self.expand_and_clean(data) for data in sorted_data_table
        ]
        return expanded_data_table

    def get_data_idx(self):
        return {d["region_id"]: d for d in self.get_data_table()}

    def has_values(self) -> bool:
        return True
