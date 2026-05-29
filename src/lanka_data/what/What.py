class What:
    pass

    @classmethod
    def get_aggr_data(cls, data_list):
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

    def get_result(self, regions) -> list[dict]:
        data_list = self.get_data_list(regions)
        aggr_data = self.get_aggr_data(data_list)
        source_info = self.get_source_info()

        return (
            dict(
                data_list=data_list,
                aggr_data=aggr_data,
            )
            | source_info
        )
