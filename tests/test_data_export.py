import json
import os

import pandas as pd

from lanka_data.command.Command import Command
from lanka_data.visual.data_export.ChartSpecVisual import ChartSpecVisual
from lanka_data.visual.data_export.CSVVisual import CSVVisual
from lanka_data.visual.data_export.GeoJSONVisual import GeoJSONVisual
from lanka_data.visual.data_export.ParquetVisual import ParquetVisual
from lanka_data.visual.data_export.TableVisual import TableVisual
from lanka_data.visual.data_export.TSVVisual import TSVVisual
from lanka_data.visual.VisualFactory import VisualFactory


class FakeDataset:
    def get_data_table(self):
        return [
            {
                "region_id": "LK-1",
                "region_name": "Western",
                "center_lat": 6.9,
                "center_lng": 80.0,
                "values": {"Buddhist": 30, "Hindu": 10},
                "total_value": 40,
            },
            {
                "region_id": "LK-2",
                "region_name": "Central",
                "center_lat": 7.3,
                "center_lng": 80.6,
                "values": {"Hindu": 50, "Buddhist": 5},
                "total_value": 55,
            },
        ]

    def get_sources(self):
        return []


def build(how_cmd):
    command = Command.from_str(f"Religion/2024/LK:province/{how_cmd}")
    visual = VisualFactory.from_command_and_datasets(command, [FakeDataset()])
    return visual


def read_text(result):
    with open(result["file_path"], encoding="utf-8") as fin:
        return fin.read()


class TestDataExport:
    def test_factory_resolves_export_visuals(self):
        assert isinstance(build("CSV"), CSVVisual)
        assert isinstance(build("TSV"), TSVVisual)
        assert isinstance(build("Table"), TableVisual)
        assert isinstance(build("GeoJSON"), GeoJSONVisual)
        assert isinstance(build("Parquet"), ParquetVisual)
        assert isinstance(build("ChartSpec"), ChartSpecVisual)

    def test_build_returns_only_file_link(self):
        result = build("CSV").build()
        assert set(result.keys()) == {"file_path"}
        assert os.path.exists(result["file_path"])

    def test_csv_orders_categories_by_total(self):
        text = read_text(build("CSV").build())
        lines = text.strip().split("\n")
        assert lines[0] == (
            "region_id,region_name,Hindu,Buddhist,total_value"
        )
        assert lines[1] == "LK-1,Western,10,30,40"
        assert lines[2] == "LK-2,Central,50,5,55"

    def test_tsv_uses_tab_delimiter(self):
        text = read_text(build("TSV").build())
        header = text.strip().split("\n")[0]
        assert (
            header == "region_id\tregion_name\tHindu\tBuddhist\ttotal_value"
        )

    def test_table_renders_markdown_grid(self):
        lines = read_text(build("Table").build()).split("\n")
        assert lines[0].startswith("| region_id | region_name |")
        assert set(lines[1]) <= {"|", "-"}
        assert "| LK-1" in lines[2]

    def test_missing_category_defaults_to_zero(self):
        text = read_text(build("CSV").build())
        assert "LK-2,Central,50,5,55" in text

    def test_geojson_builds_feature_collection(self):
        geojson = json.loads(read_text(build("GeoJSON").build()))
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 2
        feature = geojson["features"][0]
        assert feature["geometry"] == {
            "type": "Point",
            "coordinates": [80.0, 6.9],
        }
        assert feature["properties"]["region_id"] == "LK-1"
        assert feature["properties"]["total_value"] == 40

    def test_parquet_roundtrips_table(self):
        frame = pd.read_parquet(build("Parquet").build()["file_path"])
        assert list(frame.columns) == [
            "region_id",
            "region_name",
            "Hindu",
            "Buddhist",
            "total_value",
        ]
        assert len(frame) == 2

    def test_chart_spec_lists_categories(self):
        spec = json.loads(read_text(build("ChartSpec").build()))
        assert spec["schema"] == "lanka-data-chart-spec/v1"
        assert spec["categories"] == ["Hindu", "Buddhist"]
        assert len(spec["data"]) == 2
