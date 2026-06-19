from lanka_data.dataset.Dataset import Dataset


class ValueDataset(Dataset):
    def __init__(self, region_ids):
        Dataset.__init__(self)
        self.region_ids = region_ids

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
        return sorted_data_table
