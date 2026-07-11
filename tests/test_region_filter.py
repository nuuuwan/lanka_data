import pytest

from lanka_data.api.dataset.RegionValueDataset.RegionValueDataset import \
    RegionValueDataset
from lanka_data.api.fields.How import How
from lanka_data.api.fields.RegionFilter import RegionFilter
from lanka_data.api.fields.Where import Where as APIWhere
from lanka_data.command.UnknownHowError import UnknownHowError
from lanka_data.dataset.DatasetFactory import DatasetFactory


def _rows():
    return [
        {
            "region_id": f"r{i}",
            "total_value": i * 10,
            "pct_values": {"Buddhist": i / 10.0},
        }
        for i in range(1, 12)
    ]


class TestRegionFilter:
    def test_parses_rank_modifier(self):
        region_filter = RegionFilter.from_modifier("Top10")
        assert region_filter.kind == "rank"
        assert region_filter.direction == "Top"
        assert region_filter.count == 10

    def test_parses_threshold_modifier(self):
        region_filter = RegionFilter.from_modifier("Buddhist>0.9")
        assert region_filter.kind == "threshold"
        assert region_filter.category == "Buddhist"
        assert region_filter.op == ">"
        assert region_filter.threshold == 0.9

    def test_returns_none_for_plain_modifier(self):
        assert RegionFilter.from_modifier("Buddhist") is None
        assert RegionFilter.from_modifier("") is None
        assert RegionFilter.from_modifier(None) is None

    def test_apply_top_keeps_highest_values(self):
        kept = RegionFilter.from_modifier("Top3").apply(_rows())
        assert [row["region_id"] for row in kept] == ["r11", "r10", "r9"]

    def test_apply_bottom_keeps_lowest_values(self):
        kept = RegionFilter.from_modifier("Bottom2").apply(_rows())
        assert [row["region_id"] for row in kept] == ["r1", "r2"]

    def test_apply_threshold_filters_by_share(self):
        kept = RegionFilter.from_modifier("Buddhist>0.9").apply(_rows())
        assert [row["region_id"] for row in kept] == ["r10", "r11"]

    def test_apply_threshold_supports_less_than_or_equal(self):
        kept = RegionFilter.from_modifier("Buddhist<=0.2").apply(_rows())
        assert [row["region_id"] for row in kept] == ["r1", "r2"]


class TestHowRegionFilter:
    def test_rank_modifier_accepted_on_non_category_base(self):
        how = How("BarChart:Top10")
        assert how.region_filter.kind == "rank"
        assert how.category is None

    def test_threshold_modifier_strips_category(self):
        how = How("Map:Buddhist>0.9")
        assert how.category == "Buddhist"
        assert how.region_filter.kind == "threshold"

    def test_plain_category_has_no_region_filter(self):
        how = How("Map:Buddhist")
        assert how.region_filter is None
        assert how.category == "Buddhist"

    def test_invalid_modifier_still_rejected(self):
        with pytest.raises(UnknownHowError):
            How("BarChart:Nope")

    def test_format_labels_filters(self):
        assert How("BarChart:Top10").format() == "Bar Chart (Top 10)"
        assert How("Map:Buddhist>0.9").format() == "Buddhist > 0.9"


class _FakeDataset(RegionValueDataset):
    def __init__(self, region_data_list, source_rows):
        RegionValueDataset.__init__(self, region_data_list)
        self._source_rows = source_rows

    def get_year(self):
        return "Y"

    def get_sources(self):
        return []

    def get_source_data_table(self):
        return self._source_rows

    def clean_data_row(self, row):
        return row


def _build_fake_dataset():
    region_data_list = []
    source_rows = []
    for i in range(1, 12):
        region_id = f"r{i}"
        region_data_list.append(
            {
                "region_id": region_id,
                "region_name": region_id,
                "center_lat": 0,
                "center_lng": 0,
                "current_ids": [region_id],
            }
        )
        source_rows.append(
            {"region_id": region_id, "values": {"Buddhist": i, "Other": 1}}
        )
    return _FakeDataset(region_data_list, source_rows)


class TestDatasetRegionFilter:
    def test_get_data_table_without_filter_returns_all(self):
        dataset = _build_fake_dataset()
        assert len(dataset.get_data_table()) == 11

    def test_get_data_table_applies_rank_filter(self):
        dataset = _build_fake_dataset()
        dataset.region_filter = RegionFilter.from_modifier("Top3")
        region_ids = [row["region_id"] for row in dataset.get_data_table()]
        assert region_ids == ["r11", "r10", "r9"]

    def test_get_data_table_applies_threshold_filter(self):
        dataset = _build_fake_dataset()
        dataset.region_filter = RegionFilter.from_modifier("Buddhist>0.9")
        region_ids = {row["region_id"] for row in dataset.get_data_table()}
        assert region_ids == {"r10", "r11"}

    def test_get_data_table_empty_when_no_region_matches(self):
        dataset = _build_fake_dataset()
        dataset.region_filter = RegionFilter.from_modifier("Buddhist>0.99")
        assert dataset.get_data_table() == []

    def test_get_category_keys_ignores_region_filter(self):
        dataset = _build_fake_dataset()
        dataset.region_filter = RegionFilter.from_modifier("Buddhist>0.99")
        assert "Buddhist" in dataset.get_category_keys()


class _FakeCommand:
    def __init__(self, how, where):
        self.how = How(how)
        self.where = APIWhere(where)


class TestWhereTopRegionFilter:
    def test_where_top_applies_rank_filter(self):
        dataset = _build_fake_dataset()
        command = _FakeCommand("BarChart", "LK:rivers#3")
        DatasetFactory._with_region_filter(dataset, command)
        region_ids = [row["region_id"] for row in dataset.get_data_table()]
        assert region_ids == ["r11", "r10", "r9"]

    def test_how_filter_takes_priority_over_where(self):
        dataset = _build_fake_dataset()
        command = _FakeCommand("BarChart:Top2", "LK:rivers#5")
        DatasetFactory._with_region_filter(dataset, command)
        region_ids = [row["region_id"] for row in dataset.get_data_table()]
        assert region_ids == ["r11", "r10"]
