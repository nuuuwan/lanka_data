from lanka_data.api.fields.How import How
from lanka_data.visual.plot_visual.BivariateMapVisual import (
    BivariateData,
    BivariateMapVisual,
    BivariatePalette,
    QuadrantMapVisual,
)
from lanka_data.visual.VisualFactory import VisualFactory


def _row(region_id, pct1, pct2):
    return {
        "region_id": region_id,
        "region_name": region_id,
        "pct_values1": pct1,
        "pct_values2": pct2,
    }


class TestBivariateRouting:
    BASES = {
        "BivariateMap": BivariateMapVisual,
        "QuadrantMap": QuadrantMapVisual,
    }

    def test_bases_are_registered(self):
        for base in self.BASES:
            assert base in How.BASE_LABELS

    def test_factory_maps_each_base(self):
        for base, cls in self.BASES.items():
            assert VisualFactory._VISUAL_CLS[base] is cls

    def test_bases_have_labels(self):
        for base in self.BASES:
            assert How(base).format()

    def test_bases_are_not_category_or_interval(self):
        for base in self.BASES:
            assert base not in How.CATEGORY_BASES
            assert base not in How.INTERVAL_BASES

    def test_bases_accept_category_pair_modifier(self):
        for base in self.BASES:
            how = How(f"{base}:Buddhist:Sinhalese")
            assert how.categories == ["Buddhist", "Sinhalese"]
            assert how.category is None


class TestBivariateData:
    def test_dominant_returns_top_share(self):
        label, share = BivariateData._dominant({"A": 0.7, "B": 0.3})
        assert label == "A" and share == 0.7

    def test_dominant_handles_empty(self):
        assert BivariateData._dominant({}) == (None, 0.0)

    def test_point_uses_both_measures(self):
        point = BivariateData._point(
            _row("LK-1", {"Buddhist": 0.6, "Hindu": 0.4}, {"Sinhala": 0.8})
        )
        assert point["x_label"] == "Buddhist"
        assert point["y_label"] == "Sinhala"
        assert point["x"] == 0.6 and point["y"] == 0.8

    def test_point_uses_explicit_categories(self):
        point = BivariateData._point(
            _row(
                "LK-1",
                {"Buddhist": 0.6, "Hindu": 0.4},
                {"Sinhalese": 0.7, "Tamil": 0.3},
            ),
            ["Hindu", "Tamil"],
        )
        assert point["x_label"] == "Hindu"
        assert point["y_label"] == "Tamil"
        assert point["x"] == 0.4 and point["y"] == 0.3

    def test_point_missing_category_share_is_zero(self):
        point = BivariateData._point(
            _row("LK-1", {"Buddhist": 1.0}, {"Sinhalese": 1.0}),
            ["Islam", "Tamil"],
        )
        assert point["x"] == 0.0 and point["y"] == 0.0

    def test_point_skips_when_a_measure_is_missing(self):
        assert BivariateData._point(_row("LK-1", {}, {"S": 1.0})) is None

    def test_points_skips_incomplete_rows(self):
        table = [
            _row("LK-1", {"A": 1.0}, {"S": 1.0}),
            {"region_id": "LK-2", "pct_values1": {"A": 1.0}},
        ]
        assert len(BivariateData.points(table)) == 1

    def test_thresholds_split_into_bins(self):
        assert BivariateData.thresholds([0, 1, 2, 3, 4, 5], 3) == [2, 4]

    def test_thresholds_empty_when_too_few_bins(self):
        assert BivariateData.thresholds([1, 2, 3], 1) == []

    def test_bin_index_counts_crossed_thresholds(self):
        assert BivariateData.bin_index(2.5, [2, 4]) == 1
        assert BivariateData.bin_index(4, [2, 4]) == 2
        assert BivariateData.bin_index(0, [2, 4]) == 0

    def test_classify_assigns_extreme_bins(self):
        table = [
            _row("LK-1", {"A": 0.5, "B": 0.5}, {"S": 0.5, "T": 0.5}),
            _row("LK-2", {"A": 0.9, "B": 0.1}, {"S": 0.9, "T": 0.1}),
            _row("LK-3", {"A": 0.7, "B": 0.3}, {"S": 0.7, "T": 0.3}),
        ]
        points = BivariateData.classify(BivariateData.points(table), 3)
        by_id = {p["region_id"]: p for p in points}
        assert by_id["LK-1"]["x_bin"] == 0
        assert by_id["LK-2"]["x_bin"] == 2


class TestBivariatePalette:
    def test_three_bin_grid_is_square(self):
        palette = BivariatePalette(3)
        assert palette.color(0, 0) != palette.color(2, 2)

    def test_two_bin_grid_used_for_quadrant(self):
        palette = BivariatePalette(2)
        assert palette.grid is BivariatePalette.GRID_2

    def test_out_of_range_bins_are_clamped(self):
        palette = BivariatePalette(3)
        assert palette.color(9, 9) == palette.color(2, 2)

    def test_none_bin_is_neutral(self):
        assert BivariatePalette(3).color(None, 1) == BivariatePalette.NEUTRAL
