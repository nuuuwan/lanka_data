from lanka_data.api.data.DataSource import DataSource
from lanka_data.api.dataset.RegionValueDataset.RegionValueDataset import \
    RegionValueDataset
from utils_future import Log

log = Log("SeriesDataset")


class SeriesDataset(RegionValueDataset):
    def __init__(self, year_labels, datasets):
        RegionValueDataset.__init__(self, datasets[0].region_data_list)
        self.year_labels = year_labels
        self.datasets = datasets

    def get_year(self):
        return f"{self.year_labels[0]} \u2192 {self.year_labels[-1]}"

    def __str__(self):
        return f"SeriesDataset({', '.join(self.year_labels)})"

    def get_sources(self) -> list:
        return DataSource.merge_datasource_list_of_lists(
            [dataset.get_sources() for dataset in self.datasets]
        )

    def has_values(self) -> bool:
        return all(dataset.has_values() for dataset in self.datasets)

    def _get_union_region_ids(self, data_idx_list):
        region_ids = set()
        for data_idx in data_idx_list:
            region_ids |= set(data_idx.keys())
        return sorted(region_ids)

    def _build_region_row(self, region_id, data_idx_list):
        year_values = {}
        latest = {}
        for year_label, data_idx in zip(self.year_labels, data_idx_list):
            data = data_idx.get(region_id, {})
            values = data.get("values", {})
            year_values[year_label] = values
            if values:
                latest = data
        return dict(
            region_id=region_id,
            region_name=latest.get("region_name", str(region_id)),
            center_lat=latest.get("center_lat"),
            center_lng=latest.get("center_lng"),
            current_ids=latest.get("current_ids", [region_id]),
            values=latest.get("values", {}),
            total_value=latest.get("total_value", 0),
            pct_values=latest.get("pct_values", {}),
            year_values=year_values,
        )

    def get_data_table(self):
        data_idx_list = [dataset.get_data_idx() for dataset in self.datasets]
        return self._apply_region_filter(
            [
                self._build_region_row(region_id, data_idx_list)
                for region_id in self._get_union_region_ids(data_idx_list)
            ]
        )

    def get_source_data_table(self) -> list[dict]:
        raise NotImplementedError

    def clean_data_row(self, row: dict) -> dict:
        raise NotImplementedError
