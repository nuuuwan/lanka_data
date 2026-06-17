from lanka_data.api.what.What import What
from utils_future import Log

log = Log("DiffWhat")


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

    # flake8: noqa: CFQ001
    @staticmethod
    def _remap_data(data1, data2):
        values1 = data1["values"]
        values2 = data2["values"]
        pct_values1 = data1["pct_values"]
        pct_values2 = data2["pct_values"]

        keys1 = set(values1.keys())
        keys2 = set(values2.keys())
        keys_union = keys1.union(keys2)

        values = {}
        p_values = {}
        change1_sum = 0
        for k in keys_union:
            value1 = values1.get(k, 0)
            value2 = values2.get(k, 0)
            value_change = round(value1 - value2, 0)
            values[k] = value_change

            pct_value1 = pct_values1.get(k, 0)
            pct_value2 = pct_values2.get(k, 0)
            pct_change = round(pct_value1 - pct_value2, 4)
            p_values[k] = pct_change

            change1_sum += abs(pct_change)

        change = round(change1_sum / len(keys_union), 4) if keys_union else 0

        values = dict(sorted(values.items(), key=lambda item: -item[1]))
        total_value = sum(values.values())
        p_values = dict(sorted(p_values.items(), key=lambda item: -item[1]))

        data = {}
        for k, v in data1.items():
            if k not in ["values", "pct_values"]:
                data[k] = v

        max1 = list(values1.keys())[0] if values1 else '(No Data)'
        max2 = list(values2.keys())[0] if values2 else '(No Data)'

        data |= dict(
            values1=values1,
            values2=values2,
            pct_values1=pct_values1,
            pct_values2=pct_values2,
            values=values,
            pct_values=p_values,
            total_value=total_value,
            change=change,
            max1=max1,
            max2=max2,
            flip=f'{max1} to {max2}' if max1 != max2 else '(No Flip)',
        )

        return data

    def get_data_list(self, where):
        data_list1 = self.what1.get_data_list(where)
        data_list2 = self.what2.get_data_list(where)

        data_idx1 = {data["region_id"]: data for data in data_list1}
        data_idx2 = {data["region_id"]: data for data in data_list2}

        region_ids1 = set(data_idx1.keys())
        region_ids2 = set(data_idx2.keys())
        diff1_2 = region_ids1.difference(region_ids2)
        diff2_1 = region_ids2.difference(region_ids1)
        if diff1_2 or diff2_1:
            log.warning(
                "what1 and what2 should have the same set of region_ids."
                + f" {diff1_2} in what1 not in what2."
                + f" {diff2_1} in what2 not in what1"
            )
        common_ids = region_ids1.intersection(region_ids2)
        combined_data_list = []
        for id in common_ids:
            data1 = self._remap_data(data_idx1[id], data_idx2[id])
            combined_data_list.append(data1)

        combined_data_list.sort(key=lambda d: d["region_id"])
        return combined_data_list

    def get_source_info(self):
        return self.what1.get_source_info()

    @classmethod
    def get_aggregated_value_data(cls, data_list):
        return {}
