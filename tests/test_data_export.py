from lanka_data.command.Command import Command
from lanka_data.visual.data_export.CSVVisual import CSVVisual
from lanka_data.visual.data_export.TableVisual import TableVisual
from lanka_data.visual.data_export.TSVVisual import TSVVisual
from lanka_data.visual.VisualFactory import VisualFactory


class FakeDataset:
    def get_data_table(self):
        return [
            {
                "region_id": "LK-1",
                "region_name": "Western",
                "values": {"Buddhist": 30, "Hindu": 10},
                "total_value": 40,
            },
            {
                "region_id": "LK-2",
                "region_name": "Central",
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


class TestDataExport:
    def test_factory_resolves_export_visuals(self):
        assert isinstance(build("CSV"), CSVVisual)
        assert isinstance(build("TSV"), TSVVisual)
        assert isinstance(build("Table"), TableVisual)

    def test_csv_orders_categories_by_total(self):
        text = build("CSV").build()
        lines = text.strip().split("\n")
        assert lines[0] == (
            "region_id,region_name,Hindu,Buddhist,total_value"
        )
        assert lines[1] == "LK-1,Western,10,30,40"
        assert lines[2] == "LK-2,Central,50,5,55"

    def test_tsv_uses_tab_delimiter(self):
        text = build("TSV").build()
        header = text.strip().split("\n")[0]
        assert (
            header == "region_id\tregion_name\tHindu\tBuddhist\ttotal_value"
        )

    def test_table_renders_markdown_grid(self):
        lines = build("Table").build().split("\n")
        assert lines[0].startswith("| region_id | region_name |")
        assert set(lines[1]) <= {"|", "-"}
        assert "| LK-1" in lines[2]

    def test_missing_category_defaults_to_zero(self):
        text = build("CSV").build()
        assert "LK-2,Central,50,5,55" in text
