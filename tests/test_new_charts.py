from lanka_data.api.fields.How import How
from lanka_data.visual.plot_visual.HistogramVisual import HistogramData
from lanka_data.visual.plot_visual.ScatterPlotVisual import ScatterPlotData
from lanka_data.visual.plot_visual.StackedBarChartVisual import (
    StackedBarChartVisual,
)
from lanka_data.visual.plot_visual.TreeMapVisual import (
    TreeMapData,
    TreeMapVisual,
)
from lanka_data.visual.plot_visual.HistogramVisual import HistogramVisual
from lanka_data.visual.plot_visual.ScatterPlotVisual import ScatterPlotVisual
from lanka_data.visual.VisualFactory import VisualFactory


class TestNewChartRouting:
    BASES = {
        "StackedBarChart": StackedBarChartVisual,
        "TreeMap": TreeMapVisual,
        "Histogram": HistogramVisual,
        "ScatterPlot": ScatterPlotVisual,
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


class TestStackedBarChartNormalize:
    def test_each_region_sums_to_one(self):
        subregions = [{"values": {"A": 30, "B": 70}}]
        out = StackedBarChartVisual._normalize_subregions(subregions)
        assert abs(sum(out[0]["values"].values()) - 1.0) < 1e-9

    def test_preserves_ratio(self):
        subregions = [{"values": {"A": 25, "B": 75}}]
        out = StackedBarChartVisual._normalize_subregions(subregions)
        assert abs(out[0]["values"]["A"] - 0.25) < 1e-9

    def test_zero_values_are_dropped(self):
        subregions = [{"values": {"A": 0, "B": 5}}]
        out = StackedBarChartVisual._normalize_subregions(subregions)
        assert "A" not in out[0]["values"]

    def test_all_zero_does_not_crash(self):
        subregions = [{"values": {"A": 0}}]
        out = StackedBarChartVisual._normalize_subregions(subregions)
        assert out[0]["values"] == {}

    def test_percent_formatter(self):
        assert StackedBarChartVisual._format_millions(0.25, None) == "25%"


class TestTreeMapData:
    RECT = (0, 0, 4, 3)

    @staticmethod
    def _area(rect):
        return rect[2] * rect[3]

    def test_areas_are_proportional_to_values(self):
        rects = TreeMapData.layout([3, 1], *self.RECT)
        assert abs(self._area(rects[0]) / self._area(rects[1]) - 3) < 1e-6

    def test_total_area_is_covered(self):
        rects = TreeMapData.layout([2, 2, 1], *self.RECT)
        total = sum(self._area(r) for r in rects)
        assert abs(total - self._area((0, 0, 4, 3))) < 1e-6

    def test_rects_stay_within_bounds(self):
        rects = TreeMapData.layout([5, 3, 2, 1], *self.RECT)
        for x, y, dx, dy in rects:
            assert x >= -1e-9 and y >= -1e-9
            assert x + dx <= 4 + 1e-9
            assert y + dy <= 3 + 1e-9

    def test_count_matches_positive_values(self):
        rects = TreeMapData.layout([5, 0, 3, -2], *self.RECT)
        assert len(rects) == 2

    def test_empty_values_give_no_rects(self):
        assert TreeMapData.layout([], *self.RECT) == []

    def test_category_totals_aggregate_across_regions(self):
        subregions = [
            {"values": {"A": 10, "B": 5}},
            {"values": {"A": 2, "B": 8}},
        ]
        totals = TreeMapVisual._category_totals(subregions)
        assert totals == {"A": 12, "B": 13}


class TestHistogramData:
    def test_counts_sum_to_number_of_values(self):
        _, counts = HistogramData.bins([1, 2, 3, 4, 5], 2)
        assert sum(counts) == 5

    def test_edges_span_min_to_max(self):
        edges, _ = HistogramData.bins([10, 20, 30], 4)
        assert edges[0] == 10 and edges[-1] == 30

    def test_max_value_falls_in_last_bin(self):
        _, counts = HistogramData.bins([0, 10], 5)
        assert counts[-1] == 1

    def test_single_value_does_not_crash(self):
        edges, counts = HistogramData.bins([7], 3)
        assert sum(counts) == 1

    def test_empty_values_give_empty(self):
        assert HistogramData.bins([], 5) == ([], [])


class TestScatterPlotData:
    def test_point_has_total_and_top_share(self):
        subregions = [
            {
                "region_name": "R",
                "total_value": 100,
                "values": {"A": 75, "B": 25},
            }
        ]
        points = ScatterPlotData.points(subregions)
        assert points == [(100, 0.75, "A", "R")]

    def test_total_defaults_to_value_sum(self):
        subregions = [
            {"region_name": "R", "values": {"A": 3, "B": 1}}
        ]
        total, share, label, name = ScatterPlotData.points(subregions)[0]
        assert total == 4 and label == "A"

    def test_regions_without_positive_values_are_skipped(self):
        subregions = [{"region_name": "R", "values": {"A": 0}}]
        assert ScatterPlotData.points(subregions) == []
