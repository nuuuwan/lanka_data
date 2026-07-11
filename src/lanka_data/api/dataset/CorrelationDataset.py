from lanka_data.api.data.DataSource import DataSource
from lanka_data.api.dataset.RegionValueDataset.RegionValueDataset import \
    RegionValueDataset


class CorrelationDataset(RegionValueDataset):
    JOIN_DELIM = " & "

    def __init__(
        self, dataset1: RegionValueDataset, dataset2: RegionValueDataset
    ):
        RegionValueDataset.__init__(self, dataset1.region_data_list)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def get_year(self):
        return (
            "Correlation between"
            + f" {self.dataset1.get_year()} and {self.dataset2.get_year()}"
        )

    def __str__(self):
        return f"CorrelationDataset({self.dataset1} x {self.dataset2})"

    def get_sources(self) -> list:
        return DataSource.merge_datasource_list_of_lists(
            [self.dataset1.get_sources(), self.dataset2.get_sources()]
        )

    def has_values(self):
        return self.dataset1.has_values() and self.dataset2.has_values()

    @classmethod
    def _join_key(cls, category1, category2):
        return f"{category1}{cls.JOIN_DELIM}{category2}"

    @classmethod
    def _compute_joint(cls, total, pct_values1, pct_values2):
        values = {}
        pct_values = {}
        for category1, share1 in pct_values1.items():
            for category2, share2 in pct_values2.items():
                key = cls._join_key(category1, category2)
                pct = share1 * share2
                pct_values[key] = round(
                    pct, RegionValueDataset.PCT_VALUE_PRECISION
                )
                values[key] = int(round(total * pct))
        return values, pct_values

    @classmethod
    def _compute_region_correlation(cls, data1, data2):
        pct_values1 = data1.get("pct_values", {})
        pct_values2 = data2.get("pct_values", {})
        total = data1.get("total_value", 0)
        values, pct_values = cls._compute_joint(
            total, pct_values1, pct_values2
        )
        return dict(
            region_id=data1["region_id"],
            region_name=data1["region_name"],
            center_lat=data1["center_lat"],
            center_lng=data1["center_lng"],
            current_ids=data1["current_ids"],
            values1=data1.get("values", {}),
            pct_values1=pct_values1,
            total_value1=total,
            values2=data2.get("values", {}),
            pct_values2=pct_values2,
            total_value2=data2.get("total_value", 0),
            values=dict(sorted(values.items(), key=lambda item: -item[1])),
            total_value=sum(values.values()),
            pct_values=pct_values,
        )

    def get_data_table(self):
        data_idx1 = self.dataset1.get_data_idx()
        data_idx2 = self.dataset2.get_data_idx()
        common_region_ids = set(self.dataset1.region_ids).intersection(
            set(self.dataset2.region_ids)
        )
        return self._apply_region_filter(
            [
                self._compute_region_correlation(
                    data_idx1[region_id], data_idx2[region_id]
                )
                for region_id in sorted(common_region_ids)
            ]
        )

    def get_source_data_table(self) -> list[dict]:
        raise NotImplementedError

    def clean_data_row(self, row: dict) -> dict:
        raise NotImplementedError
