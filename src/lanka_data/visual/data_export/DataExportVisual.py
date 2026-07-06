from lanka_data.visual.Visual import Visual


class DataExportVisual(Visual):
    def _get_data_table(self):
        return self.datasets[-1].get_data_table()

    @staticmethod
    def _category_labels(data_table):
        totals = {}
        for row in data_table:
            for category, value in (row.get("values") or {}).items():
                totals[category] = totals.get(category, 0) + value
        return [
            category
            for category, _ in sorted(
                totals.items(), key=lambda item: -item[1]
            )
        ]

    @staticmethod
    def _row_total(row):
        values = row.get("values") or {}
        total = row.get("total_value")
        if total is None:
            total = sum(values.values())
        return total

    def get_table(self):
        data_table = self._get_data_table()
        categories = self._category_labels(data_table)
        headers = ["region_id", "region_name"] + categories + ["total_value"]
        rows = []
        for row in data_table:
            values = row.get("values") or {}
            rows.append(
                [row.get("region_id"), row.get("region_name")]
                + [values.get(category, 0) for category in categories]
                + [self._row_total(row)]
            )
        return headers, rows
