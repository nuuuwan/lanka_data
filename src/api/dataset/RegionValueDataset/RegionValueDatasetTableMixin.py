from api.utils_future import Log

log = Log("RegionValueDataset")


class RegionValueDatasetTableMixin:
    def _collect_data_for_region(
        self, region_id, current_ids, complete_data_idx
    ):
        data_list = [
            complete_data_idx[cid]
            for cid in current_ids
            if cid in complete_data_idx
        ]
        if not data_list:
            log.warning(
                f"No data found for region_id={region_id} "
                f"with current_ids={current_ids}"
            )
        return data_list

    @staticmethod
    def _aggregate_region_values(data_list):
        values = {}
        for data in data_list:
            for k, v in data["values"].items():
                values[k] = values.get(k, 0) + v
        return dict(sorted(values.items(), key=lambda item: -item[1]))

    @staticmethod
    def _validate_no_pre_ids(current_ids):
        for current_id in current_ids:
            if "-pre" in current_id:
                raise ValueError(f"Invalid current_id: {current_id}")

    def _build_region_entry(self, region_id, current_ids, values):
        region = self.get_region(region_id)
        return dict(
            region_id=region_id,
            region_name=region["region_name"],
            center_lat=region["center_lat"],
            center_lng=region["center_lng"],
            current_ids=current_ids,
            values=values,
        )

    def get_data_table(self):
        complete_data_table = self.get_complete_data_table()
        complete_data_idx = {d["region_id"]: d for d in complete_data_table}
        filtered_data_table = []
        for region_id, current_ids in self.region_id_to_current_ids.items():
            data_list = self._collect_data_for_region(
                region_id, current_ids, complete_data_idx
            )
            self._validate_no_pre_ids(current_ids)
            values = self._aggregate_region_values(data_list)
            filtered_data_table.append(
                self._build_region_entry(region_id, current_ids, values)
            )
        sorted_data_table = sorted(
            filtered_data_table, key=lambda row: row["region_id"]
        )
        expanded_data_table = [
            self.expand_and_clean(data) for data in sorted_data_table
        ]
        if not expanded_data_table:
            raise ValueError(
                "No data available for the specified regions. "
                "Please check the region IDs and data source."
            )
        return expanded_data_table

    def get_data_idx(self):
        return {d["region_id"]: d for d in self.get_data_table()}
