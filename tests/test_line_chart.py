import pytest

from lanka_data.api.dataset.SeriesDataset import SeriesDataset
from lanka_data.command.Command import Command
from lanka_data.command.fields.How import How
from lanka_data.command.InvalidCommandError import InvalidCommandError
from lanka_data.dataset.DatasetFactory import DatasetFactory
from lanka_data.visual.plot_visual.LineChartVisual import LineChartVisual


class _FakeYearDataset:
    region_data_list = [{"region_id": "LK-1"}]

    def __init__(self, year, data_idx):
        self.year = year
        self._data_idx = data_idx

    def get_data_idx(self):
        return self._data_idx

    def get_sources(self):
        return []

    def has_values(self):
        return True


class TestLineChart:
    def test_line_chart_how_needs_interval_and_series(self):
        how = How("LineChart")
        assert how.needs_interval
        assert how.needs_series
        assert how.format() == "Line Chart"

    def test_line_chart_requires_interval_when(self):
        with pytest.raises(InvalidCommandError):
            Command.from_str("Religion/2024/LK-1/LineChart")

    def test_interval_when_builds_series_for_line_chart(self, monkeypatch):
        calls = []

        def fake_from_command(command):
            calls.append(command.when_cmd)
            return _FakeYearDataset(command.when_cmd, {})

        monkeypatch.setattr(DatasetFactory, "from_command", fake_from_command)
        command = Command.from_str("Religion/2001-2012-2024/LK-1/LineChart")
        datasets = DatasetFactory.list_from_command(command)
        assert calls == ["2001", "2012", "2024"]
        assert len(datasets) == 1
        assert isinstance(datasets[0], SeriesDataset)
        assert datasets[0].year_labels == ["2001", "2012", "2024"]
        assert not datasets[0].is_diff()

    def test_series_dataset_exposes_year_values(self):
        d1 = _FakeYearDataset(
            "2001",
            {
                "LK-1": {
                    "region_id": "LK-1",
                    "region_name": "A",
                    "values": {"X": 10, "Y": 5},
                    "total_value": 15,
                    "pct_values": {},
                }
            },
        )
        d2 = _FakeYearDataset(
            "2024",
            {
                "LK-1": {
                    "region_id": "LK-1",
                    "region_name": "A",
                    "values": {"X": 20, "Y": 2},
                    "total_value": 22,
                    "pct_values": {},
                }
            },
        )
        series = SeriesDataset(["2001", "2024"], [d1, d2])
        row = series.get_data_table()[0]
        assert row["values"] == {"X": 20, "Y": 2}
        assert row["year_values"]["2001"] == {"X": 10, "Y": 5}
        assert row["year_values"]["2024"] == {"X": 20, "Y": 2}

    def test_aggregate_series_sums_across_regions(self):
        data_table = [
            {"year_values": {"2001": {"X": 10}, "2024": {"X": 20}}},
            {"year_values": {"2001": {"X": 1}, "2024": {"X": 2}}},
        ]
        series = LineChartVisual._aggregate_series(
            data_table, ["2001", "2024"], ["X"]
        )
        assert series == {"X": [11, 22]}
        assert LineChartVisual._select_series(series, ["X"]) == ["X"]

    def test_select_series_drops_all_zero_and_caps(self):
        series = {"X": [0, 0], "Y": [1, 2]}
        assert LineChartVisual._select_series(series, ["X", "Y"]) == ["Y"]
