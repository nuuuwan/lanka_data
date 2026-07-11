import json

from lanka_data.visual.data_export.DataExportVisual import DataExportVisual
from lanka_data.visual.data_export.FileExportMixin import FileExportMixin


class ChartSpecVisual(FileExportMixin, DataExportVisual):
    SCHEMA = "lanka-data-chart-spec/v1"

    @classmethod
    def get_description(cls):
        return "Exports data as chart specification in JSON format"

    def build(self):
        data_table = self._get_data_table()
        spec = {
            "schema": self.SCHEMA,
            "title": self.command.cmd_id,
            "mark": "bar",
            "encoding": {
                "x": "region_name",
                "y": "value",
                "series": "category",
            },
            "categories": self._category_labels(data_table),
            "data": [self._to_datum(row) for row in data_table],
        }
        return self._write_output(
            "ChartSpec.json", json.dumps(spec, indent=2)
        )

    def _to_datum(self, row):
        return {
            "region_id": row.get("region_id"),
            "region_name": row.get("region_name"),
            "values": row.get("values") or {},
            "total_value": self._row_total(row),
        }
