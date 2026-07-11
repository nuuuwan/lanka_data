import json

from lanka_data.visual.data_export.DataExportVisual import DataExportVisual
from lanka_data.visual.data_export.FileExportMixin import FileExportMixin


class GeoJSONVisual(FileExportMixin, DataExportVisual):
    @classmethod
    def get_description(cls):
        return "Exports data as GeoJSON format with region geometries and properties"

    def build(self):
        feature_collection = {
            "type": "FeatureCollection",
            "features": [
                self._to_feature(row) for row in self._get_data_table()
            ],
        }
        content = json.dumps(feature_collection, indent=2)
        return self._write_output("Data.geojson", content)

    def _to_feature(self, row):
        return {
            "type": "Feature",
            "geometry": self._to_geometry(row),
            "properties": {
                "region_id": row.get("region_id"),
                "region_name": row.get("region_name"),
                "values": row.get("values") or {},
                "total_value": self._row_total(row),
            },
        }

    @staticmethod
    def _to_geometry(row):
        lat = row.get("center_lat")
        lng = row.get("center_lng")
        if lat is None or lng is None:
            return None
        return {"type": "Point", "coordinates": [lng, lat]}
