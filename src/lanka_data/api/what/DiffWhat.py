from lanka_data.api.what.What import What


class DiffWhat(What):
    def __init__(self, what1: What, what2: What):
        if what1.title != what2.title:
            raise ValueError(
                "what1 and what2 should have the same title,"
                + f" but got {what1.title} and {what2.title}"
            )
        super().__init__(what1.title)
        self.what1 = what1
        self.what2 = what2

    def get_description(self):
        return self.what1.get_description()

    @staticmethod
    def _remap_data(data1, data2):
        values1 = data1["values"]
        values2 = data2["values"]
        data1["values1"] = values1
        data1["values2"] = values2
        values = {}
        keys1 = set(values1.keys())
        keys2 = set(values2.keys())
        keys_union = keys1.union(keys2)
        pct_values = {}
        for k in keys_union:
            v1 = values1.get(k, 0)
            v2 = values2.get(k, 0)
            values[k] = v1 - v2
            pct_values[k] = round((v1 - v2) / v2, 4) if v2 else 0
        values = dict(sorted(values.items(), key=lambda item: -item[1]))
        total_value = sum(values.values())
        pct_values = dict(
            sorted(pct_values.items(), key=lambda item: -item[1])
        )

        data = {}
        for k, v in data1.items():
            if k not in ("values", "total_value", "pct_values"):
                data[k] = v

        data["values"] = values
        data["total_value"] = total_value
        data["pct_values"] = pct_values

        return data

    def get_data_list(self, where):
        data_list1 = self.what1.get_data_list(where)
        data_list2 = self.what2.get_data_list(where)

        data_idx1 = {data["region_id"]: data for data in data_list1}
        data_idx2 = {data["region_id"]: data for data in data_list2}

        region_ids1 = set(data_idx1.keys())
        region_ids2 = set(data_idx2.keys())
        if region_ids1 != region_ids2:
            raise ValueError(
                "Region ids do not match between what1 and what2 for"
                + f" where: {where}. Region ids in what1: {region_ids1},"
                + f" region ids in what2: {region_ids2}"
            )

        combined_data_list = []
        for id in region_ids1:
            data1 = self._remap_data(data_idx1[id], data_idx2[id])
            combined_data_list.append(data1)

        combined_data_list.sort(key=lambda d: d["region_id"])
        return combined_data_list

    def get_source_info(self):
        return self.what1.get_source_info()

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

        return dict(
            values=aggr_values,
            total_value=total_value,
        )
