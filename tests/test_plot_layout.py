import pytest

from lanka_data.visual.plot.PlotLayout import PlotLayout
from lanka_data.visual.plot.PlotLayoutError import PlotLayoutError


class TestPlotLayout:
    def test_one_dataset_is_square(self):
        layout = PlotLayout(1)
        assert (layout.n_cols, layout.n_rows) == (1, 1)
        assert layout.figsize == (9, 9)

    def test_two_datasets_is_two_by_one(self):
        layout = PlotLayout(2)
        assert (layout.n_cols, layout.n_rows) == (2, 1)
        assert layout.figsize == (18, 9)

    def test_three_datasets_is_three_by_one(self):
        layout = PlotLayout(3)
        assert (layout.n_cols, layout.n_rows) == (3, 1)
        assert layout.figsize == (27, 9)

    def test_four_datasets_is_two_by_two(self):
        layout = PlotLayout(4)
        assert (layout.n_cols, layout.n_rows) == (2, 2)
        assert layout.figsize == (18, 18)

    def test_positions_are_row_major(self):
        layout = PlotLayout(4)
        positions = [layout.position(i) for i in range(4)]
        assert positions == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_more_than_four_raises(self):
        with pytest.raises(PlotLayoutError):
            PlotLayout(5)

    def test_zero_raises(self):
        with pytest.raises(PlotLayoutError):
            PlotLayout(0)
