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
