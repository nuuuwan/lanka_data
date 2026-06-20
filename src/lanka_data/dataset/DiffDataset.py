from lanka_data.dataset.Dataset import Dataset
from lanka_data.dataset.RegionValueDataset import RegionValueDataset


class DiffDataset(Dataset):
    def __init__(
        self, dataset1: RegionValueDataset, dataset2: RegionValueDataset
    ):
        Dataset.__init__(self)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __str__(self):
        return f"DiffDataset({self.dataset1} - {self.dataset2})"

    def is_diff(self):
        return True

    def get_source_info_list(self) -> list:
        return (
            self.dataset1.get_source_info_list()
            + self.dataset2.get_source_info_list()
        )

    def has_values(self):
        return self.dataset1.has_values() and self.dataset2.has_values()

    def get_data_table(self):
        data_idx1 = self.dataset1.get_data_idx()
        data_idx2 = self.dataset2.get_data_idx()

        region_ids1 = self.dataset1.region_ids
        region_ids2 = self.dataset2.region_ids
        common_region_ids = list(
            sorted(set(region_ids1).intersection(set(region_ids2)))
        )

        d_list = []
        for region_id in common_region_ids:
            data1 = data_idx1[region_id]
            data2 = data_idx2[region_id]

            values1 = data1["values"]
            values2 = data2["values"]
            pct_values1 = data1["pct_values"]
            pct_values2 = data2["pct_values"]
            common_keys = set(values1.keys()).intersection(
                set(values2.keys())
            )

            values = {}
            pct_values = {}
            for k in common_keys:
                values[k] = values2[k] - values1[k]
                pct_values[k] = pct_values2[k] - pct_values1[k]
            total_value = sum(values.values())

            d = dict(
                region_id=data1["region_id"],
                region_name=data1["region_name"],
                values1=values1,
                pct_values1=pct_values1,
                total_value1=data1["total_value"],
                values2=values2,
                pct_values2=pct_values2,
                total_value2=data2["total_value"],
                values=values,
                total_value=total_value,
                pct_values=pct_values,
            )
            d_list.append(d)
        return d_list
