from lanka_data.command.Command import Command
from lanka_data.dataset.custom.Census2012Dataset import Census2012Dataset
from lanka_data.dataset.DatasetFactory import DatasetFactory


class TestDatasetFactory:
    def test_census_dataset_resolves_by_declared_registration(
        self, monkeypatch
    ):
        sentinel = object()
        monkeypatch.setattr(
            DatasetFactory,
            "get_region_data_list",
            lambda command: [{"region_id": "LK"}],
        )
        monkeypatch.setattr(
            Census2012Dataset,
            "from_label_and_region_data_list",
            lambda label, data: sentinel,
        )
        command = Command.from_str("Religion/2012/LK/JSON")
        assert DatasetFactory.from_command(command) is sentinel

    def test_interval_when_builds_only_diff(self, monkeypatch):
        calls = []

        class DummyDataset:
            region_data_list = []

            def __init__(self, year):
                self.year = year

            def get_year(self):
                return self.year

        def fake_from_command(command):
            calls.append(command.when_cmd)
            return DummyDataset(command.when_cmd)

        monkeypatch.setattr(DatasetFactory, "from_command", fake_from_command)
        command = Command.from_str("Religion/2012-2024/LK/JSON")
        datasets = DatasetFactory.list_from_command(command)
        assert calls == ["2012", "2024"]
        assert len(datasets) == 1
        assert datasets[0].is_diff()

    def test_multi_year_interval_builds_only_diff(self, monkeypatch):
        calls = []

        class DummyDataset:
            region_data_list = []

            def __init__(self, year):
                self.year = year

            def get_year(self):
                return self.year

        def fake_from_command(command):
            calls.append(command.when_cmd)
            return DummyDataset(command.when_cmd)

        monkeypatch.setattr(DatasetFactory, "from_command", fake_from_command)
        command = Command.from_str("Religion/2001-2012-2024/LK/JSON")
        datasets = DatasetFactory.list_from_command(command)
        assert calls == ["2001", "2024"]
        assert len(datasets) == 1
        assert datasets[0].is_diff()
        assert datasets[0].dataset1.year == "2001"
        assert datasets[0].dataset2.year == "2024"

    def test_combined_what_builds_only_correlation(self, monkeypatch):
        calls = []

        class DummyDataset:
            region_data_list = []
            region_ids = []

            def __init__(self, what):
                self.what = what

            def get_year(self):
                return self.what

        def fake_from_command(command):
            calls.append(command.what_cmd)
            return DummyDataset(command.what_cmd)

        monkeypatch.setattr(DatasetFactory, "from_command", fake_from_command)
        command = Command.from_str("Religion+Ethnicity/2024/LK:province/Map")
        datasets = DatasetFactory.list_from_command(command)
        assert calls == ["Religion", "Ethnicity"]
        assert len(datasets) == 1
        assert not datasets[0].is_diff()
        assert datasets[0].panel_label == "Correlation: Religion & Ethnicity"
        assert datasets[0].dataset1.what == "Religion"
        assert datasets[0].dataset2.what == "Ethnicity"
