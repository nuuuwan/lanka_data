from lanka_data.visual.plot_visual.HistogramVisual.HistogramVisual import (
    HistogramVisual,
)
import matplotlib

from lanka_data.visual.plot_visual.ScatterPlotVisual.ScatterPlotData import (
    ScatterPlotData,
)
from lanka_data.visual.plot_visual.ScatterPlotVisual.ScatterPlotVisual import (
    ScatterPlotVisual,
)
from lanka_data.visual.plot.color_spec.ClusterData import ClusterData
from lanka_data.visual.plot.color_spec.ColorSpec.ColorSpec import ColorSpec
from lanka_data.visual.plot.color_spec.ColorSpecHelpers.ColorSpecHelpers import (
    ColorSpecHelpers,
)
from lanka_data.visual.plot_visual.StackedBarChartVisual.StackedBarChartVisual import (
    StackedBarChartVisual,
)
from lanka_data.visual.plot_visual.TreeMapVisual.TreeMapData import (
    TreeMapData,
)
from lanka_data.visual.plot_visual.TreeMapVisual.TreeMapVisual import (
    TreeMapVisual,
)
from lanka_data.visual.VisualFactory import VisualFactory

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lanka_data.api.fields.How import How  # noqa: E402
from lanka_data.visual.plot_visual.BarChartVisual.BarChartVisual import (
    BarChartVisual,
)  # noqa: E402
from lanka_data.visual.plot_visual.HistogramVisual.HistogramData import (
    HistogramData,
)  # noqa: E402


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


class _FakeDataset:
    def __init__(self, rows):
        self._rows = rows

    def get_data_table(self):
        return self._rows


class _BarChartForTest(BarChartVisual):
    def __init__(self):
        self.how_cmd = None

    def _build_category_to_color(self, dataset, category_labels):
        return {c: "C%d" % (i % 9) for i, c in enumerate(category_labels)}


class TestBarChartSingleVsStacked:
    SINGLE = [
        {
            "region_id": "LK-42",
            "region_name": "Gampaha",
            "values": {"Buddhist": 100, "Hindu": 30, "Muslim": 20},
        }
    ]
    MULTI = [
        {
            "region_id": "LK-41",
            "region_name": "Colombo",
            "values": {"Buddhist": 80, "Hindu": 40, "Muslim": 30},
        },
        {
            "region_id": "LK-42",
            "region_name": "Gampaha",
            "values": {"Buddhist": 100, "Hindu": 30, "Muslim": 20},
        },
    ]
    RIVERS = [
        {
            "region_id": "R-1",
            "region_name": "Mahaweli Ganga",
            "values": {"RiverLen": 330},
        },
        {
            "region_id": "R-2",
            "region_name": "Malwathu Oya",
            "values": {"RiverLen": 161},
        },
    ]

    @staticmethod
    def _draw(rows):
        fig = plt.figure()
        _BarChartForTest().draw(_FakeDataset(rows), fig)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        texts = [t.get_text() for t in ax.texts]
        plt.close(fig)
        return labels, texts

    def test_single_row_uses_category_x_axis(self):
        labels, _ = self._draw(self.SINGLE)
        assert labels == ["Buddhist", "Hindu", "Muslim"]

    def test_multi_row_labels_regions_on_x_axis(self):
        labels, _ = self._draw(self.MULTI)
        assert "Colombo" in labels
        assert "Gampaha" in labels

    def test_multi_category_bars_show_percent(self):
        _, texts = self._draw(self.MULTI)
        assert any("%" in t for t in texts)

    def test_single_category_bars_show_unit_not_percent(self):
        _, texts = self._draw(self.RIVERS)
        assert not any("%" in t for t in texts)
        assert any("330 km" == t for t in texts)


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
        subregions = [{"region_name": "R", "values": {"A": 3, "B": 1}}]
        total, share, label, name = ScatterPlotData.points(subregions)[0]
        assert total == 4 and label == "A"

    def test_regions_without_positive_values_are_skipped(self):
        subregions = [{"region_name": "R", "values": {"A": 0}}]
        assert ScatterPlotData.points(subregions) == []


class TestClusterData:
    def test_labels_match_number_of_values(self):
        labels, _ = ClusterData.cluster([1, 2, 100, 101], 2)
        assert len(labels) == 4

    def test_similar_values_share_a_cluster(self):
        labels, _ = ClusterData.cluster([1, 2, 100, 101], 2)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_centers_count_capped_by_distinct_values(self):
        _, centers = ClusterData.cluster([5, 5, 5], 4)
        assert len(centers) == 1

    def test_single_value_does_not_crash(self):
        labels, centers = ClusterData.cluster([7], 3)
        assert labels == [0] and len(centers) == 1

    def test_empty_values_give_empty(self):
        assert ClusterData.cluster([], 3) == ([], [])


class TestClusterModifier:
    def test_map_cluster_is_recognized(self):
        how = How("Map:Cluster-3")
        assert how.is_cluster
        assert how.cluster_n == 3

    def test_hexmap_cluster_is_recognized(self):
        how = How("HexMap:Cluster-4")
        assert how.is_cluster
        assert how.cluster_n == 4

    def test_cluster_without_count_uses_default(self):
        assert How("Map:Cluster").cluster_n == 5

    def test_cluster_is_not_a_category(self):
        assert How("Map:Cluster-3").category is None

    def test_plain_map_is_not_a_cluster(self):
        how = How("Map")
        assert not how.is_cluster
        assert how.cluster_n is None

    def test_cluster_routes_to_map_visual(self):
        from lanka_data.visual.plot_visual.MapVisual import MapVisual

        assert (
            VisualFactory._VISUAL_CLS[How("Map:Cluster-3").base] is MapVisual
        )

    def test_cluster_is_not_a_standalone_base(self):
        assert "Cluster" not in How.BASE_LABELS
        assert "Cluster" not in VisualFactory._VISUAL_CLS


class TestClusterColorSpec:
    ROWS = [
        {
            "region_id": "LK-1",
            "region_name": "A",
            "values": {"Sinhalese": 5, "SLMoor": 3, "Islam": 2},
            "pct_values": {"Sinhalese": 0.5, "SLMoor": 0.3, "Islam": 0.2},
            "total_value": 10,
        },
        {
            "region_id": "LK-2",
            "region_name": "B",
            "values": {"Sinhalese": 6, "SLMoor": 3, "Islam": 2},
            "pct_values": {"Sinhalese": 0.5, "SLMoor": 0.3, "Islam": 0.2},
            "total_value": 11,
        },
        {
            "region_id": "LK-3",
            "region_name": "C",
            "values": {"SLTamil": 60, "Sinhalese": 40},
            "pct_values": {"SLTamil": 0.6, "Sinhalese": 0.4},
            "total_value": 100,
        },
        {
            "region_id": "LK-4",
            "region_name": "D",
            "values": {"SLTamil": 61, "Sinhalese": 40},
            "pct_values": {"SLTamil": 0.6, "Sinhalese": 0.4},
            "total_value": 101,
        },
    ]

    def _spec(self, n):
        return ColorSpecHelpers.get_color_spec_for_cluster(
            _FakeDataset(self.ROWS), n
        )

    def test_every_region_is_coloured(self):
        spec = self._spec(2)
        assert set(spec.region_to_color) == {"LK-1", "LK-2", "LK-3", "LK-4"}

    def test_similar_regions_share_a_colour(self):
        spec = self._spec(2)
        assert spec.region_to_color["LK-1"] == spec.region_to_color["LK-2"]
        assert spec.region_to_color["LK-3"] == spec.region_to_color["LK-4"]
        assert spec.region_to_color["LK-1"] != spec.region_to_color["LK-3"]

    def test_colour_matches_dominant_field(self):
        spec = self._spec(2)
        sinhalese = ColorSpec.cmap_for_label("Sinhalese")(1.0)
        r, g, b, alpha = spec.region_to_color["LK-1"]
        assert (r, g, b) == (sinhalese[0], sinhalese[1], sinhalese[2])
        assert alpha == 0.5

    def test_legend_label_lists_top_two_fields_and_other(self):
        spec = self._spec(2)
        labels = set(spec.value_to_color)
        assert "Sinhalese (50%), SLMoor (30%), Other (20%)" in labels
        assert "SLTamil (60%), Sinhalese (40%), Other (0%)" in labels

    def test_empty_dataset_does_not_crash(self):
        spec = ColorSpecHelpers.get_color_spec_for_cluster(
            _FakeDataset([]), 3
        )
        assert spec.region_to_color == {}
