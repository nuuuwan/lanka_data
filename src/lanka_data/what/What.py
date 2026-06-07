class What:
    def __init__(self, title):
        self.title = title

    def get_title(self) -> str:
        return self.title

    @classmethod
    def get_aggregated_value_data(cls, data_list):
        aggr_values = {}
        for data in data_list:
            values = data["values"]
            for k, v in values.items():
                aggr_values[k] = aggr_values.get(k, 0) + v

        aggr_values = dict(
            sorted(aggr_values.items(), key=lambda item: -item[1])
        )
        total_value = sum(aggr_values.values())
        pct_values = {
            k: round(v / total_value, 4) for k, v in aggr_values.items()
        }
        return dict(
            values=aggr_values,
            total_value=total_value,
            pct_values=pct_values,
        )

    @classmethod
    def get_values(cls, data):
        return data.get("values")

    @classmethod
    def get_where_to_what_id_map(cls, regions) -> dict:
        idx = {}
        for region in regions.raw_region_data_list:
            region_id = region["region_id"]
            current_ids = region.get("current_ids", [region_id])
            idx[region_id] = current_ids
        return idx

    @classmethod
    def _get_mapped_data_for_region(
        cls, region_id, current_ids, source_data_idx, raw_region_data_idx
    ):
        source_data_list_for_region = []
        for current_id in current_ids:
            data_for_current = source_data_idx.get(current_id)
            if not data_for_current:
                raise ValueError(
                    f"No data found for region_id={current_id}"
                    + f" (mapped from {region_id})."
                )
            source_data_list_for_region.append(data_for_current)

        if not source_data_list_for_region:
            raise ValueError(
                f"No data found for region_id={region_id}"
                + f" with current_ids={current_ids}."
            )

        source_data_list_for_region_with_values = [
            cls.extract_source_data_values(d)
            for d in source_data_list_for_region
        ]
        aggr_data = cls.get_aggregated_value_data(
            source_data_list_for_region_with_values
        )
        source_data_for_region = raw_region_data_idx[region_id]
        mapped_data = (
            dict(
                region_id=region_id,
                region_name=source_data_for_region["region_name"],
            )
            | raw_region_data_idx[region_id]
            | aggr_data
        )
        return mapped_data

    def get_data_list(self, regions) -> list[dict]:
        source_data_list = self.get_source_data_list()
        source_data_idx = {
            d["region_id"]: d for d in source_data_list if "region_id" in d
        }

        raw_region_data_idx = {
            r["region_id"]: r for r in regions.raw_region_data_list
        }

        mapped_data_list = []
        for region_id, current_ids in self.get_where_to_what_id_map(
            regions
        ).items():
            mapped_data = self._get_mapped_data_for_region(
                region_id, current_ids, source_data_idx, raw_region_data_idx
            )
            mapped_data_list.append(mapped_data)

        return mapped_data_list
