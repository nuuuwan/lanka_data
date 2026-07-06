from lanka_data.api.dataset.CorrelationDataset import CorrelationDataset


class _DummyDataset:
    def __init__(self, data_idx):
        self.region_data_list = [
            {"region_id": region_id} for region_id in data_idx
        ]
        self.region_ids = list(data_idx)
        self._data_idx = data_idx

    def get_data_idx(self):
        return self._data_idx

    def get_year(self):
        return "Y"

    def get_sources(self):
        return []

    def has_values(self):
        return True


def _region_data(region_id, values, pct_values, total_value):
    return {
        "region_id": region_id,
        "region_name": region_id,
        "center_lat": 0,
        "center_lng": 0,
        "current_ids": [region_id],
        "values": values,
        "pct_values": pct_values,
        "total_value": total_value,
    }


class TestCorrelationDataset:
    def _build(self):
        idx1 = {
            "LK-1": _region_data(
                "LK-1",
                {"Buddhist": 60, "Hindu": 40},
                {"Buddhist": 0.6, "Hindu": 0.4},
                100,
            )
        }
        idx2 = {
            "LK-1": _region_data(
                "LK-1",
                {"Sinhala": 70, "Tamil": 30},
                {"Sinhala": 0.7, "Tamil": 0.3},
                100,
            )
        }
        return CorrelationDataset(_DummyDataset(idx1), _DummyDataset(idx2))

    def test_correlation_is_not_a_diff(self):
        assert self._build().is_diff() is False

    def test_correlation_builds_joint_distribution(self):
        row = self._build().get_data_table()[0]
        assert row["values"]["Buddhist & Sinhala"] == 42
        assert row["values"]["Hindu & Tamil"] == 12
        assert row["pct_values"]["Buddhist & Tamil"] == 0.18
        assert row["total_value"] == 100

    def test_correlation_year_label_mentions_both(self):
        assert self._build().get_year() == "Correlation between Y and Y"
