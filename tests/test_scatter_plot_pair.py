import matplotlib

from lanka_data.api.fields.How import How
from lanka_data.visual.plot_visual.ScatterPlotVisual.ScatterPlotStats import ScatterPlotStats
from lanka_data.visual.plot_visual.ScatterPlotVisual.ScatterPlotVisual import ScatterPlotVisual
from lanka_data.visual.VisualFactory import VisualFactory

matplotlib.use("Agg")


class TestScatterPlotPairRegistry:
    def test_scatter_plot_is_pair_category_base(self):
        assert "ScatterPlot" in How.PAIR_CATEGORY_BASES

    def test_plus_separated_categories_are_parsed(self):
        how = How("ScatterPlot:Buddhist+Sinhalese")
        assert how.categories == ["Buddhist", "Sinhalese"]
        assert how.category is None

    def test_colon_separated_categories_are_parsed(self):
        how = How("ScatterPlot:Buddhist:Sinhalese")
        assert how.categories == ["Buddhist", "Sinhalese"]

    def test_factory_maps_scatter_plot(self):
        assert VisualFactory._VISUAL_CLS["ScatterPlot"] is ScatterPlotVisual


class TestScatterPlotStats:
    def test_perfect_line_has_unit_correlation(self):
        fit = ScatterPlotStats.fit([0, 1, 2, 3], [0.0, 0.1, 0.2, 0.3])
        assert round(fit["slope"], 3) == 0.1
        assert round(fit["intercept"], 3) == 0.0
        assert round(fit["r"], 3) == 1.0
        assert round(fit["r2"], 3) == 1.0
        assert fit["n"] == 4

    def test_negative_correlation(self):
        fit = ScatterPlotStats.fit([0, 1, 2, 3], [3, 2, 1, 0])
        assert fit["slope"] < 0
        assert round(fit["r"], 3) == -1.0

    def test_single_point_has_no_fit(self):
        assert ScatterPlotStats.fit([1], [2]) is None

    def test_zero_variance_has_no_fit(self):
        assert ScatterPlotStats.fit([1, 1, 1], [1, 2, 3]) is None

    def test_line_endpoints_span_x_range(self):
        fit = ScatterPlotStats.fit([0, 1, 2, 3], [0.0, 0.1, 0.2, 0.3])
        xs, ys = ScatterPlotStats.line_endpoints(fit, [0, 1, 2, 3])
        assert xs == [0, 3]
        assert round(ys[0], 3) == 0.0 and round(ys[1], 3) == 0.3

    def test_text_reports_all_stats(self):
        text = ScatterPlotStats.text(ScatterPlotStats.fit([0, 1], [0, 1]))
        assert "y = " in text
        assert "r = " in text
        assert "R\u00b2 = " in text
        assert "n = 2" in text
