from lanka_data.dataset.Dataset import Dataset


class RegionValueDataset(Dataset):
    PCT_VALUE_PRECISION = 4

    def __init__(self, region_ids):
        Dataset.__init__(self)
        self.region_ids = region_ids

    def expand(self, data):
        total_value = data["total_value"]
        pct_values = {
            k: round(v / total_value, self.PCT_VALUE_PRECISION)
            for k, v in data["values"].items()
        }
        data["pct_values"] = pct_values
        return data

    def get_data_table(self):
        complete_data_table = self.get_complete_data_table()
        filtered_data_table = [
            row
            for row in complete_data_table
            if row["region_id"] in self.region_ids
        ]
        sorted_data_table = sorted(
            filtered_data_table,
            key=lambda row: self.region_ids.index(row["region_id"]),
        )
        expanded_data_table = [
            self.expand(data) for data in sorted_data_table
        ]
        return expanded_data_table
