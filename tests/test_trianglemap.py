import math

from lanka_data.visual.plot.map.TriangleData.TriangleData import TriangleData
from lanka_data.visual.plot_visual.TriangleMapVisual.TriangleMapVisual import (
    TriangleMapVisual,
)
from lanka_data.visual.plot_visual.TriangleMapVisual.TriangleMapBoundaryMixin \
    import TriangleMapBoundaryMixin
from lanka_data.visual.plot_visual.TriangleMapVisual.TriangleTextFit import (
    TriangleTextFit,
)
from lanka_data.visual.VisualFactory import VisualFactory


class TestTriangleDataPure:
    def test_counts_are_population_proportional(self):
        counts = TriangleData.get_counts({"A": 900, "B": 100})
        assert counts["A"] > counts["B"]
        assert min(counts.values()) >= 1

    def test_build_grid_has_enough_cells(self):
        centers, size = TriangleData.build_grid((0, 0, 10, 10), 20)
        assert len(centers) >= 20
        assert size > 0

    def test_grid_cells_carry_orientation(self):
        centers, _ = TriangleData.build_grid((0, 0, 10, 10), 20)
        orientations = {up for _, _, up in centers}
        assert orientations == {True, False}

    def test_assign_gives_each_slot_a_distinct_cell(self):
        centroids = {"A": (0, 0), "B": (10, 10)}
        counts = {"A": 2, "B": 1}
        centers, _ = TriangleData.build_grid((0, 0, 10, 10), 3)
        triangles = TriangleData.assign(centroids, counts, centers)
        assert len(triangles) == 3
        used = {(x, y, up) for _, x, y, up in triangles}
        assert len(used) == 3
        region_ids = [region_id for region_id, _, _, _ in triangles]
        assert region_ids.count("A") == 2
        assert region_ids.count("B") == 1


class TestTriangleMapRouting:
    def test_trianglemap_base_is_registered(self):
        from lanka_data.api.fields.How import How

        assert "TriangleMap" in How.BASE_LABELS
        assert "TriangleMap" in How.CATEGORY_BASES

    def test_factory_maps_trianglemap_to_trianglemap_visual(self):
        assert VisualFactory._VISUAL_CLS["TriangleMap"] is TriangleMapVisual


class TestTriangleScaleRange:
    def test_range_reflects_lowest_and_highest_per_region(self):
        region_to_weight = {"A": 100, "B": 100}
        triangles = [
            ("A", 0, 0, True),
            ("A", 1, 0, False),
            ("B", 2, 0, True),
        ]
        value_min, value_max = TriangleData._value_per_triangle_range(
            region_to_weight, triangles
        )
        assert value_min == 50
        assert value_max == 100

    def test_empty_triangles_gives_no_range(self):
        assert TriangleData._value_per_triangle_range({}, []) == (None, None)

    def test_scale_text_shows_range_when_values_differ(self):
        text = TriangleMapVisual._scale_text(100, 300)
        assert text == "Triangle = 200 ± 100 people"

    def test_scale_text_collapses_when_values_equal(self):
        text = TriangleMapVisual._scale_text(100, 100)
        assert text == "Triangle = 100 people"


class TestTriangleBoundary:
    SIZE = 1.0

    def test_up_and_down_triangles_share_an_edge(self):
        up = TriangleMapBoundaryMixin._triangle_polygon(
            0.5, 0.0, self.SIZE, True
        )
        down = TriangleMapBoundaryMixin._triangle_polygon(
            1.0, math.sqrt(3) / 6, self.SIZE, False
        )
        merged = TriangleMapBoundaryMixin._merge([up, down], self.SIZE)
        assert merged.geom_type == "Polygon"

    def test_draw_boundaries_plots_region_ring(self):
        layout = {
            "size": self.SIZE,
            "triangles": [["A", 0.5, 0.0, True]],
        }
        drawn = []

        class FakeAx:
            def plot(self, xs, ys, **kwargs):
                drawn.append((list(xs), list(ys)))

        TriangleMapBoundaryMixin._draw_boundaries(FakeAx(), layout)
        assert len(drawn) == 1


class TestTriangleTextFit:
    SIZE = 1.0

    def _fit(self, points):
        return TriangleTextFit.best_label_fit(points, self.SIZE)

    def test_horizontal_run_is_centered_and_horizontal(self):
        points = [(col, 0.0) for col in range(4)]
        cx, cy, width, _, angle = self._fit(points)
        assert angle == 0.0
        assert cx == 1.5
        assert cy == 0.0
        assert width == 3 + self.SIZE

    def test_single_triangle_defaults_to_its_center(self):
        cx, cy, width, _, angle = self._fit([(5.0, 7.0)])
        assert (cx, cy) == (5.0, 7.0)
        assert width == self.SIZE
        assert angle == 0.0
