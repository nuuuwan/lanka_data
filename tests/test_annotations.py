from lanka_data.visual.annotations.Annotations import Annotations
from lanka_data.visual.annotations.NumberAbbreviator import NumberAbbreviator
from lanka_data.visual.plot.Caption import Caption
from lanka_data.visual.plot.color_spec.ColorSpec.ColorSpec import ColorSpec


PLAIN = [
    {
        "region_id": "LK-1",
        "region_name": "Western",
        "values": {"A": 30, "B": 10},
        "total_value": 40,
    },
    {
        "region_id": "LK-2",
        "region_name": "Central",
        "values": {"A": 5},
        "total_value": 5,
    },
    {
        "region_id": "LK-3",
        "region_name": "North",
        "values": {"A": 45},
        "total_value": 45,
    },
]

DIFF = [
    {
        "region_id": "LK-1",
        "region_name": "Western",
        "values": {"A": -30},
        "total_value": -30,
        "change": 0.1,
        "values1": {"A": 1},
    },
    {
        "region_id": "LK-2",
        "region_name": "Central",
        "values": {"A": 500},
        "total_value": 500,
        "change": 0.2,
        "values1": {"A": 1},
    },
]


class TestNumberAbbreviator:
    def test_millions(self):
        assert NumberAbbreviator.format(2_500_000) == "2.5M"

    def test_thousands(self):
        assert NumberAbbreviator.format(1_200) == "1.2K"

    def test_negative(self):
        assert NumberAbbreviator.format(-2_000_000) == "-2.0M"

    def test_signed_positive(self):
        assert NumberAbbreviator.signed(500) == "+500"

    def test_signed_negative(self):
        assert NumberAbbreviator.signed(-30) == "-30"


class TestAnnotationsExtremes:
    def test_highest_and_lowest(self):
        callouts = Annotations.from_data_table(PLAIN).callouts()
        assert callouts[0] == "Highest: North (45)"
        assert callouts[1] == "Lowest: Central (5.00)"

    def test_summary_prefix(self):
        summary = Annotations.from_data_table(PLAIN).summary()
        assert summary.startswith("What to notice —")

    def test_empty_table_has_no_callouts(self):
        assert Annotations.from_data_table([]).callouts() == []
        assert Annotations.from_data_table([]).summary() == ""

    def test_single_region_has_no_extremes(self):
        one = [PLAIN[0]]
        assert Annotations.from_data_table(one).callouts() == []

    def test_total_value_inferred_from_values(self):
        rows = [
            {"region_name": "A", "values": {"x": 1}},
            {"region_name": "B", "values": {"x": 9}},
        ]
        callouts = Annotations.from_data_table(rows).callouts()
        assert "Highest: B" in callouts[0]


class TestAnnotationsChange:
    def test_biggest_change_reported_for_diff(self):
        callouts = Annotations.from_data_table(DIFF).callouts()
        assert any(c.startswith("Biggest change: Central") for c in callouts)

    def test_change_is_signed(self):
        callouts = Annotations.from_data_table(DIFF).callouts()
        change = next(c for c in callouts if c.startswith("Biggest change"))
        assert "(+500)" in change

    def test_plain_table_has_no_change_callout(self):
        callouts = Annotations.from_data_table(PLAIN).callouts()
        assert not any(c.startswith("Biggest change") for c in callouts)


class TestAnnotationsOutliers:
    def test_outlier_flagged(self):
        pairs = [(chr(65 + i), 10) for i in range(9)] + [("Z", 100)]
        rows = [
            {"region_name": n, "values": {"x": v}, "total_value": v}
            for n, v in pairs
        ]
        callouts = Annotations.from_data_table(rows).callouts()
        assert any("Outliers: Z" in c for c in callouts)

    def test_no_outliers_when_uniform(self):
        rows = [
            {"region_name": n, "values": {"x": 10}, "total_value": 10}
            for n in ["A", "B", "C", "D"]
        ]
        callouts = Annotations.from_data_table(rows).callouts()
        assert not any(c.startswith("Outliers") for c in callouts)

    def test_fewer_than_three_rows_has_no_outliers(self):
        rows = [
            {"region_name": "A", "values": {"x": 1}, "total_value": 1},
            {"region_name": "B", "values": {"x": 1000}, "total_value": 1000},
        ]
        callouts = Annotations.from_data_table(rows).callouts()
        assert not any(c.startswith("Outliers") for c in callouts)


class FakeMapDataset:
    def __init__(self, rows):
        self._rows = rows

    def get_data_table(self):
        return self._rows

    def has_values(self):
        return True

    def is_diff(self):
        return False


class FakeVisual:
    def __init__(self, dataset, how_cmd):
        self.datasets = [dataset]
        self.how_cmd = how_cmd


MAP_ROWS = [
    {"region_id": "A", "region_name": "Alpha", "pct_values": {"X": 0.70}},
    {"region_id": "B", "region_name": "Beta", "pct_values": {"X": 0.10}},
    {"region_id": "C", "region_name": "Gamma", "pct_values": {"X": 0.40}},
]


class TestAnnotationsDisplay:
    def test_display_string_preferred_over_number(self):
        rows = [
            {"region_name": "A", "total_value": 0.7, "display": "70.0%"},
            {"region_name": "B", "total_value": 0.1, "display": "10.0%"},
        ]
        callouts = Annotations.from_data_table(rows).callouts()
        assert callouts[0] == "Highest: A (70.0%)"
        assert callouts[1] == "Lowest: B (10.0%)"


class TestColorSpecRenderedValues:
    def test_single_pct_exposes_region_values(self):
        spec = ColorSpec.by_single_pct_value(
            FakeMapDataset(MAP_ROWS),
            lambda data: data["pct_values"]["X"],
            label="X",
        )
        assert spec.region_to_value == {"A": 0.70, "B": 0.10, "C": 0.40}
        assert spec.region_to_value_str["A"] == "70.0%"

    def test_categorical_spec_has_no_region_values(self):
        spec = ColorSpec.by_custom_category_key(
            FakeMapDataset(MAP_ROWS), lambda data: "X", False
        )
        assert spec.region_to_value is None


class TestCaptionUsesVisibleData:
    def test_numeric_map_caption_uses_displayed_values(self):
        caption = Caption(FakeVisual(FakeMapDataset(MAP_ROWS), "Map:1stPct"))
        summary = caption._summary()
        assert "Highest: Alpha (70.0%)" in summary
        assert "Lowest: Beta (10.0%)" in summary

    def test_categorical_map_has_no_caption(self):
        caption = Caption(FakeVisual(FakeMapDataset(MAP_ROWS), "Map"))
        assert caption._summary() == ""
