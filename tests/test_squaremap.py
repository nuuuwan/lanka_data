import math

from lanka_data.visual.plot.map.SquareData.SquareData import SquareData
from lanka_data.visual.plot_visual.SquareMapVisual.SquareMapVisual import SquareMapVisual
from lanka_data.visual.plot_visual.SquareMapVisual.SquareMapBoundaryMixin \
    import SquareMapBoundaryMixin
from lanka_data.visual.plot_visual.SquareMapVisual.SquareTextFit import \
    SquareTextFit
from lanka_data.visual.VisualFactory import VisualFactory


class TestSquareDataPure:
    def test_counts_are_population_proportional(self):
        counts = SquareData.get_counts({"A": 900, "B": 100})
        assert counts["A"] > counts["B"]
        assert min(counts.values()) >= 1

    def test_build_grid_has_enough_cells(self):
        centers, size = SquareData.build_grid((0, 0, 10, 10), 20)
        assert len(centers) >= 20
        assert size > 0

    def test_grid_cells_are_on_a_square_lattice(self):
        centers, size = SquareData.build_grid((0, 0, 10, 10), 20)
        side = 2 * size
        xs = sorted({round(x, 6) for x, _ in centers})
        for prev, cur in zip(xs, xs[1:]):
            assert abs(cur - prev - side) < 1e-6

    def test_assign_gives_each_slot_a_distinct_cell(self):
        centroids = {"A": (0, 0), "B": (10, 10)}
        counts = {"A": 2, "B": 1}
        centers, _ = SquareData.build_grid((0, 0, 10, 10), 3)
        squares = SquareData.assign(centroids, counts, centers)
        assert len(squares) == 3
        used = {(x, y) for _, x, y in squares}
        assert len(used) == 3
        region_ids = [region_id for region_id, _, _ in squares]
        assert region_ids.count("A") == 2
        assert region_ids.count("B") == 1


class TestSquareMapRouting:
    def test_squaremap_base_is_registered(self):
        from lanka_data.api.fields.How import How

        assert "SquareMap" in How.BASE_LABELS
        assert "SquareMap" in How.CATEGORY_BASES

    def test_factory_maps_squaremap_to_squaremap_visual(self):
        assert (
            VisualFactory._VISUAL_CLS["SquareMap"] is SquareMapVisual
        )


class TestSquareScaleRange:
    def test_range_reflects_lowest_and_highest_per_region(self):
        region_to_weight = {"A": 100, "B": 100}
        squares = [("A", 0, 0), ("A", 1, 0), ("B", 2, 0)]
        value_min, value_max = SquareData._value_per_square_range(
            region_to_weight, squares
        )
        assert value_min == 50
        assert value_max == 100

    def test_empty_squares_gives_no_range(self):
        assert SquareData._value_per_square_range({}, []) == (None, None)

    def test_scale_text_shows_range_when_values_differ(self):
        text = SquareMapVisual._scale_text(100, 300)
        assert text == "Square = 200 ± 100 people"

    def test_scale_text_collapses_when_values_equal(self):
        text = SquareMapVisual._scale_text(100, 100)
        assert text == "Square = 100 people"


class TestSquareBoundary:
    SIZE = 1.0

    @classmethod
    def _grid(cls):
        side = 2 * cls.SIZE
        return [
            (col * side, row * side)
            for row in range(6)
            for col in range(6)
        ]

    def test_contiguous_region_merges_into_single_polygon(self):
        centers = self._grid()
        layout = {
            "size": self.SIZE,
            "squares": [["A", x, y] for x, y in centers],
        }
        polys = SquareMapBoundaryMixin._region_to_polygons(layout)["A"]
        merged = SquareMapBoundaryMixin._merge(polys, self.SIZE)
        assert merged.geom_type == "Polygon"
        assert len(merged.interiors) == 0

    def test_enclosed_region_creates_interior_boundary(self):
        centers = self._grid()
        side = 2 * self.SIZE
        inner = centers[14]
        ring = [
            c
            for c in centers
            if c != inner and math.dist(c, inner) < 1.5 * side
        ]
        polys = [
            SquareMapBoundaryMixin._square_polygon(x, y, self.SIZE)
            for x, y in ring
        ]
        merged = SquareMapBoundaryMixin._merge(polys, self.SIZE)
        assert merged.geom_type == "Polygon"
        assert len(merged.interiors) == 1

    def test_draw_boundaries_plots_exterior_and_interior_rings(self):
        centers = self._grid()
        side = 2 * self.SIZE
        inner = centers[14]
        ring = [
            c
            for c in centers
            if c != inner and math.dist(c, inner) < 1.5 * side
        ]
        layout = {
            "size": self.SIZE,
            "squares": [["A", x, y] for x, y in ring],
        }
        drawn = []

        class FakeAx:
            def plot(self, xs, ys, **kwargs):
                drawn.append((list(xs), list(ys)))

        SquareMapBoundaryMixin._draw_boundaries(FakeAx(), layout)
        assert len(drawn) == 2


class TestSquareTextFit:
    SIZE = 1.0
    SIDE = 2 * SIZE

    def _fit(self, points):
        return SquareTextFit.best_label_fit(points, self.SIZE)

    def test_horizontal_row_width_matches_run(self):
        points = [(col * self.SIDE, 0.0) for col in range(4)]
        cx, cy, width, height, angle = self._fit(points)
        assert angle == 0.0
        assert width == 4 * self.SIDE
        assert height == self.SIDE

    def test_center_is_midpoint_of_the_row(self):
        points = [(col * self.SIDE, 0.0) for col in range(4)]
        cx, cy, _, _, _ = self._fit(points)
        assert cx == 1.5 * self.SIDE
        assert cy == 0.0

    def test_longest_row_wins_over_shorter_ones(self):
        long_row = [(col * self.SIDE, 0.0) for col in range(5)]
        short_row = [(col * self.SIDE, self.SIDE) for col in range(2)]
        _, cy, width, _, _ = self._fit(long_row + short_row)
        assert width == 5 * self.SIDE
        assert cy == 0.0

    def test_single_square_defaults_to_its_center(self):
        cx, cy, width, _, angle = self._fit([(5.0, 7.0)])
        assert (cx, cy) == (5.0, 7.0)
        assert width == self.SIDE
        assert angle == 0.0

    def test_vertical_column_is_ninety_degrees(self):
        points = [(0.0, row * self.SIDE) for row in range(4)]
        cx, cy, width, height, angle = self._fit(points)
        assert angle == 90.0
        assert width == 4 * self.SIDE
        assert height == self.SIDE
        assert cx == 0.0
        assert cy == 1.5 * self.SIDE

    def test_longest_column_wins_over_shorter_row(self):
        column = [(0.0, row * self.SIDE) for row in range(5)]
        short_row = [(col * self.SIDE, 0.0) for col in range(2)]
        _, _, width, _, angle = self._fit(column + short_row)
        assert angle == 90.0
        assert width == 5 * self.SIDE

    def test_row_wins_tie_with_column(self):
        points = [(0.0, 0.0), (self.SIDE, 0.0), (0.0, self.SIDE)]
        _, _, _, _, angle = self._fit(points)
        assert angle == 0.0


class TestSquareContiguity:
    SIZE = 1.0

    @classmethod
    def _centers(cls, n=5):
        side = 2 * cls.SIZE
        return [
            (col * side, row * side)
            for row in range(n)
            for col in range(n)
        ]

    @classmethod
    def _components_of(cls, repaired, centers):
        adj = SquareData._adjacency(centers)
        pos = {
            (round(x, 9), round(y, 9)): i
            for i, (x, y) in enumerate(centers)
        }
        by_region = {}
        for r, x, y in repaired:
            idx = pos[(round(x, 9), round(y, 9))]
            by_region.setdefault(r, []).append(idx)
        return {
            r: SquareData._components(idxs, adj)
            for r, idxs in by_region.items()
        }

    def test_orphan_cell_is_reconnected(self):
        centers = self._centers()
        cells = [["A", 0.0, 0.0], ["A", 6.0, 0.0]]
        repaired = SquareData.repair(cells, centers)
        ids = [r for r, _, _ in repaired]
        assert ids.count("A") == 2
        comps = self._components_of(repaired, centers)
        assert len(comps["A"]) == 1

    def test_surrounded_core_reconnects_via_shift(self):
        centers = self._centers()
        cells = [
            ["A", 4.0, 4.0],
            ["A", 0.0, 0.0],
            ["B", 2.0, 4.0],
            ["B", 6.0, 4.0],
            ["B", 4.0, 2.0],
            ["B", 4.0, 6.0],
        ]
        repaired = SquareData.repair(cells, centers)
        ids = [r for r, _, _ in repaired]
        assert ids.count("A") == 2
        assert ids.count("B") == 4
        comps = self._components_of(repaired, centers)
        assert len(comps["A"]) == 1
        assert len(comps["B"]) == 1

    def test_contiguous_layout_is_unchanged(self):
        centers = self._centers()
        cells = [
            ["A", 0.0, 0.0],
            ["A", 2.0, 0.0],
            ["A", 0.0, 2.0],
        ]
        repaired = SquareData.repair(cells, centers)
        assert sorted(map(tuple, repaired)) == sorted(map(tuple, cells))

