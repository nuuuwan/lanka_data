from lanka_data.data.DataSource import DataSource
from lanka_data.dataset.RegionValueDataset import RegionValueDataset
from utils_future import Log

log = Log("DiffDataset")


class DiffDataset(RegionValueDataset):
    def __init__(
        self, dataset1: RegionValueDataset, dataset2: RegionValueDataset
    ):
        RegionValueDataset.__init__(self, dataset1.region_data_list)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def get_year(self):
        return (
            "Change between"
            + f" {self.dataset1.get_year()} and {self.dataset2.get_year()}"
        )

    def __str__(self):
        return f"DiffDataset({self.dataset1} - {self.dataset2})"

    def is_diff(self):
        return True

    def get_sources(self) -> list:
        return DataSource.merge_datasource_list_of_lists(
            [self.dataset1.get_sources(), self.dataset2.get_sources()]
        )

    def has_values(self):
        return self.dataset1.has_values() and self.dataset2.has_values()

    def get_data_table(self):
        data_idx1 = self.dataset1.get_data_idx()
        data_idx2 = self.dataset2.get_data_idx()

        region_ids1 = self.dataset1.region_ids
        region_ids2 = self.dataset2.region_ids
        union_region_ids = set(region_ids1).union(set(region_ids2))

        d_list = []
        for region_id in union_region_ids:
            data1 = data_idx1.get(region_id, {})
            data2 = data_idx2.get(region_id, {})

            values1 = data1.get("values", {})
            values2 = data2.get("values", {})
            pct_values1 = data1.get("pct_values", {})
            pct_values2 = data2.get("pct_values", {})
            common_keys = set(values1.keys()).intersection(set(values2.keys()))

            values = {}
            pct_values = {}
            error1_sum = 0
            for k in common_keys:
                values[k] = int(values2[k] - values1[k])
                pct_diff = pct_values2[k] - pct_values1[k]
                pct_values[k] = round(
                    pct_diff,
                    RegionValueDataset.PCT_VALUE_PRECISION,
                )
                error1 = abs(pct_diff)
                error1_sum += error1

            total_value = sum(values.values())

            d = dict(
                region_id=data1["region_id"],
                region_name=data1["region_name"],
                center_lat=data1["center_lat"],
                center_lng=data1["center_lng"],
                current_ids=data1["current_ids"],
                values1=values1,
                pct_values1=pct_values1,
                total_value1=data1.get("total_value", 0),
                values2=values2,
                pct_values2=pct_values2,
                total_value2=data2.get("total_value", 0),
                values=values,
                total_value=total_value,
                pct_values=pct_values,
                change=error1_sum,
            )
            d_list.append(d)
        return d_list

    def get_source_data_table(self) -> list[dict]:
        raise NotImplementedError

    def clean_data_row(self, row: dict) -> dict:
        raise NotImplementedError
