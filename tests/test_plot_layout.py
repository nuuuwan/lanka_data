import pytest

from lanka_data.visual.plot.PlotLayout import PlotLayout
from lanka_data.visual.plot.PlotLayoutError import PlotLayoutError


class TestPlotLayout:
    def test_one_dataset_is_square(self):
        layout = PlotLayout(1)
        assert (layout.n_cols, layout.n_rows) == (1, 1)
        assert layout.figsize == (9, 9)

    def test_position_is_origin(self):
        layout = PlotLayout(1)
        assert layout.position(0) == (0, 0)

    def test_two_datasets_raises(self):
        with pytest.raises(PlotLayoutError):
            PlotLayout(2)

    def test_more_than_one_raises(self):
        with pytest.raises(PlotLayoutError):
            PlotLayout(5)

    def test_zero_raises(self):
        with pytest.raises(PlotLayoutError):
            PlotLayout(0)
