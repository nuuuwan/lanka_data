class LineChartDataMixin:
    MAX_SERIES = 10

    @staticmethod
    def _get_year_labels(dataset, command):
        labels = getattr(dataset, "year_labels", None)
        if labels:
            return list(labels)
        when = getattr(command, "when", None)
        return list(when.years) if when is not None else []

    @staticmethod
    def _aggregate_series(data_table, year_labels, category_labels):
        series = {cat: [0] * len(year_labels) for cat in category_labels}
        for row in data_table:
            year_values = row.get("year_values") or {}
            for i, year_label in enumerate(year_labels):
                values = year_values.get(year_label) or {}
                for cat in category_labels:
                    series[cat][i] += values.get(cat, 0)
        return series

    @classmethod
    def _select_series(cls, series, category_labels):
        selected = [
            cat for cat in category_labels if any(v != 0 for v in series[cat])
        ]
        return selected[: cls.MAX_SERIES]
