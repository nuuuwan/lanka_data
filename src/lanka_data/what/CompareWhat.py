class CompareWhat:
    PCT_PRECISION = 4
    ERROR_PRECISION = 8

    def __init__(self, title, when_a, when_b, what_a, what_b):
        self.title = title
        self.when_a = when_a
        self.when_b = when_b
        self.what_a = what_a
        self.what_b = what_b

    def get_description(self):
        return self.what_a.get_description() or self.what_b.get_description()

    @staticmethod
    def _get_total_value(data, values):
        total_value = data.get("total_value")
        if total_value is not None:
            return total_value
        return sum(values.values())

    @classmethod
    def _get_pct_delta(cls, value_a, value_b):
        if value_b == 0:
            return 0 if value_a == 0 else None
        return round((value_a - value_b) / value_b, cls.PCT_PRECISION)

    @classmethod
    def _build_compare_data(
        cls,
        when_a,
        when_b,
        values_a,
        total_value_a,
        values_b,
        total_value_b,
    ):
        keys = sorted(
            set(values_a) | set(values_b),
            key=lambda key: (-max(values_a.get(key, 0), values_b.get(key, 0)), key),
        )
        values_by_when = {when_a: {}, when_b: {}}
        pct_values_by_when = {when_a: {}, when_b: {}}
        delta_values = {}
        pct_delta_values = {}
        error_values = {}
        total_error = 0.0

        for key in keys:
            value_a = values_a.get(key, 0)
            value_b = values_b.get(key, 0)
            pct_value_a = round(
                value_a / total_value_a, cls.PCT_PRECISION
            ) if total_value_a else 0
            pct_value_b = round(
                value_b / total_value_b, cls.PCT_PRECISION
            ) if total_value_b else 0
            pct_error = round(
                (pct_value_a - pct_value_b) ** 2, cls.ERROR_PRECISION
            )

            values_by_when[when_a][key] = value_a
            values_by_when[when_b][key] = value_b
            pct_values_by_when[when_a][key] = pct_value_a
            pct_values_by_when[when_b][key] = pct_value_b
            delta_values[key] = value_a - value_b
            pct_delta_values[key] = cls._get_pct_delta(value_a, value_b)
            error_values[key] = pct_error
            total_error += pct_error

        total_delta = total_value_a - total_value_b
        return dict(
            values_by_when=values_by_when,
            total_values_by_when={
                when_a: total_value_a,
                when_b: total_value_b,
            },
            delta_values=delta_values,
            total_delta=total_delta,
            pct_values_by_when=pct_values_by_when,
            pct_delta_values=pct_delta_values,
            total_pct_delta=cls._get_pct_delta(total_value_a, total_value_b),
            error_values=error_values,
            error=round(
                total_error / len(error_values), cls.ERROR_PRECISION
            ) if error_values else 0,
        )

    def _merge_data(self, data_a, data_b):
        values_a = self.what_a.get_values(data_a)
        values_b = self.what_b.get_values(data_b)
        if values_a is None or values_b is None:
            raise ValueError(
                f"Year comparison is not supported for '{self.title}'."
            )

        compare_data = self._build_compare_data(
            self.when_a,
            self.when_b,
            values_a,
            self._get_total_value(data_a, values_a),
            values_b,
            self._get_total_value(data_b, values_b),
        )
        excluded_keys = {
            "values",
            "total_value",
            "pct_values",
        }
        base_data = {
            key: value
            for key, value in data_a.items()
            if key not in excluded_keys
        }
        for key, value in data_b.items():
            if key in excluded_keys or key in base_data:
                continue
            base_data[key] = value
        return base_data | compare_data

    def get_data_list(self, regions):
        data_list_a = self.what_a.get_data_list(regions)
        data_list_b = self.what_b.get_data_list(regions)
        idx_a = {data["region_id"]: data for data in data_list_a}
        idx_b = {data["region_id"]: data for data in data_list_b}

        merged_data_list = []
        for region in regions.raw_region_data_list:
            region_id = region["region_id"]
            data_a = idx_a.get(region_id)
            data_b = idx_b.get(region_id)
            if data_a is None or data_b is None:
                raise ValueError(
                    f"Missing comparison data for {region_id=}."
                )
            merged_data_list.append(self._merge_data(data_a, data_b))
        return merged_data_list

    def get_source_info(self):
        source_info_a = self.what_a.get_source_info()
        source_info_b = self.what_b.get_source_info()
        return dict(
            source="Department of Census and Statistics, Sri Lanka",
            source_url="https://www.statistics.gov.lk/",
            source_by_when={
                self.when_a: source_info_a,
                self.when_b: source_info_b,
            },
        )

    def get_values(self, data):
        return data.get("delta_values")

    def get_aggregated_value_data(self, data_list):
        values_a = {}
        values_b = {}
        total_value_a = 0
        total_value_b = 0

        for data in data_list:
            region_values_a = data["values_by_when"][self.when_a]
            region_values_b = data["values_by_when"][self.when_b]
            for key, value in region_values_a.items():
                values_a[key] = values_a.get(key, 0) + value
            for key, value in region_values_b.items():
                values_b[key] = values_b.get(key, 0) + value
            total_value_a += data["total_values_by_when"][self.when_a]
            total_value_b += data["total_values_by_when"][self.when_b]

        return self._build_compare_data(
            self.when_a,
            self.when_b,
            values_a,
            total_value_a,
            values_b,
            total_value_b,
        )
