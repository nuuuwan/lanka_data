import math

from lanka_data.visual.plot.map.HexData.HexData import HexData
from lanka_data.visual.plot_visual.HexMapVisual.HexMapVisual import HexMapVisual
from lanka_data.visual.plot_visual.HexMapVisual.HexMapBoundaryMixin import \
    HexMapBoundaryMixin
from lanka_data.visual.plot_visual.HexMapVisual.HexTextFit import HexTextFit
from lanka_data.visual.VisualFactory import VisualFactory
from utils_future import HungarianUtils


class TestHungarian:
    def test_square_assignment_is_optimal(self):
        cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
        assignment = HungarianUtils.solve(cost)
        total = sum(cost[i][j] for i, j in enumerate(assignment))
        assert total == 5
        assert sorted(assignment) == [0, 1, 2]

    def test_rectangular_assignment_picks_cheapest(self):
        cost = [[10, 19, 8, 15], [10, 18, 7, 17]]
        assignment = HungarianUtils.solve(cost)
        assert len(assignment) == 2
        assert len(set(assignment)) == 2

    def test_empty_cost_returns_empty(self):
        assert HungarianUtils.solve([]) == []


class TestHexDataPure:
    def test_counts_are_population_proportional(self):
        counts = HexData.get_counts({"A": 900, "B": 100})
        assert counts["A"] > counts["B"]
        assert min(counts.values()) >= 1

    def test_build_grid_has_enough_cells(self):
        centers, radius = HexData.build_grid((0, 0, 10, 10), 20)
        assert len(centers) >= 20
        assert radius > 0

    def test_assign_gives_each_slot_a_distinct_cell(self):
        centroids = {"A": (0, 0), "B": (10, 10)}
        counts = {"A": 2, "B": 1}
        centers, _ = HexData.build_grid((0, 0, 10, 10), 3)
        hexes = HexData.assign(centroids, counts, centers)
        assert len(hexes) == 3
        used = {(x, y) for _, x, y in hexes}
        assert len(used) == 3
        region_ids = [region_id for region_id, _, _ in hexes]
        assert region_ids.count("A") == 2
        assert region_ids.count("B") == 1


class TestHexMapRouting:
    def test_hexmap_base_is_registered(self):
        from lanka_data.api.fields.How import How

        assert "HexMap" in How.BASE_LABELS

    def test_factory_maps_hexmap_to_hexmap_visual(self):
        assert VisualFactory._VISUAL_CLS["HexMap"] is HexMapVisual


class TestHexScaleMinOneHex:
    def test_extreme_skew_still_gives_each_region_one_hex(self):
        counts = HexData.get_counts({"A": 1_000_000, "B": 1, "C": 1})
        assert min(counts.values()) >= 1

    def test_every_region_appears_in_assigned_hexes(self):
        counts = HexData.get_counts({"A": 1000, "B": 100, "C": 100})
        total = sum(counts.values())
        centers, _ = HexData.build_grid((0, 0, 100, 100), total)
        centroids = {"A": (10, 10), "B": (90, 90), "C": (50, 50)}
        hexes = HexData.assign(centroids, counts, centers)
        ids = {region_id for region_id, _, _ in hexes}
        assert ids == {"A", "B", "C"}


class TestHexCountError:
    def test_no_region_error_exceeds_hexmap_error(self):
        region_to_weight = {"A": 900, "B": 250, "C": 100}
        counts = HexData.get_counts(region_to_weight)
        value_per_hex = HexData._value_per_hex(region_to_weight)
        for region_id, weight in region_to_weight.items():
            ideal = weight / value_per_hex
            error = HexData._region_error(counts[region_id], ideal)
            assert error <= HexData.HEXMAP_ERROR + 1e-9

    def test_uses_largest_population_per_hexagon(self):
        region_to_weight = {"A": 900, "B": 200, "C": 100}
        value_per_hex = HexData._value_per_hex(region_to_weight)
        smallest = min(region_to_weight.values())
        assert value_per_hex == smallest * (1 + HexData.HEXMAP_ERROR)

    def test_larger_population_per_hexagon_would_exceed_error(self):
        region_to_weight = {"A": 900, "B": 250, "C": 100}
        value_per_hex = HexData._value_per_hex(region_to_weight)
        larger = value_per_hex * 2
        smallest = min(region_to_weight.values())
        ideal = smallest / larger
        actual = max(1, round(ideal))
        assert HexData._region_error(actual, ideal) > HexData.HEXMAP_ERROR

    def test_region_error_is_absolute_percentage_error(self):
        assert HexData._region_error(4, 4.5) == abs(4 - 4.5) / 4.5

    def test_scaling_reduced_so_no_region_exceeds_error(self):
        region_to_weight = {"A": 100, "B": 175}
        value_per_hex = HexData._value_per_hex(region_to_weight)
        assert value_per_hex < min(region_to_weight.values()) * (
            1 + HexData.HEXMAP_ERROR
        )
        for weight in region_to_weight.values():
            ideal = weight / value_per_hex
            actual = max(1, round(ideal))
            error = HexData._region_error(actual, ideal)
            assert error <= HexData.HEXMAP_ERROR + 1e-9

    def test_non_positive_weights_give_each_region_one_hex(self):
        counts = HexData.get_counts({"A": 0, "B": 0})
        assert counts == {"A": 1, "B": 1}


class TestHexBoundary:
    RADIUS = 1.0

    @classmethod
    def _grid(cls):
        import math

        radius = cls.RADIUS
        dx = math.sqrt(3) * radius
        dy = 1.5 * radius
        return [
            (col * dx + (row % 2) * (dx / 2), row * dy)
            for row in range(6)
            for col in range(6)
        ]

    def test_contiguous_region_merges_into_single_polygon(self):
        centers = self._grid()
        layout = {
            "radius": self.RADIUS,
            "hexes": [["A", x, y] for x, y in centers],
        }
        polys = HexMapBoundaryMixin._region_to_polygons(layout)["A"]
        merged = HexMapBoundaryMixin._merge(polys, self.RADIUS)
        assert merged.geom_type == "Polygon"
        assert len(merged.interiors) == 0

    def test_enclosed_region_creates_interior_boundary(self):
        import math

        centers = self._grid()
        dx = math.sqrt(3) * self.RADIUS
        inner = centers[15]
        ring = [
            c
            for c in centers
            if c != inner and math.dist(c, inner) < 1.05 * dx
        ]
        polys = [
            HexMapBoundaryMixin._hex_polygon(x, y, self.RADIUS)
            for x, y in ring
        ]
        merged = HexMapBoundaryMixin._merge(polys, self.RADIUS)
        assert merged.geom_type == "Polygon"
        assert len(merged.interiors) == 1

    def test_draw_boundaries_plots_exterior_and_interior_rings(self):
        import math

        centers = self._grid()
        dx = math.sqrt(3) * self.RADIUS
        inner = centers[15]
        ring = [
            c
            for c in centers
            if c != inner and math.dist(c, inner) < 1.05 * dx
        ]
        layout = {
            "radius": self.RADIUS,
            "hexes": [["A", x, y] for x, y in ring],
        }
        drawn = []

        class FakeAx:
            def plot(self, xs, ys, **kwargs):
                drawn.append((list(xs), list(ys)))

        HexMapBoundaryMixin._draw_boundaries(FakeAx(), layout)
        assert len(drawn) == 2


class TestHexScaleRange:
    def test_range_reflects_lowest_and_highest_per_region(self):
        region_to_weight = {"A": 100, "B": 100}
        hexes = [("A", 0, 0), ("A", 1, 0), ("B", 2, 0)]
        value_min, value_max = HexData._value_per_hex_range(
            region_to_weight, hexes
        )
        assert value_min == 50
        assert value_max == 100

    def test_empty_hexes_gives_no_range(self):
        assert HexData._value_per_hex_range({}, []) == (None, None)

    def test_scale_text_shows_range_when_values_differ(self):
        text = HexMapVisual._scale_text(50, 100)
        assert text == "Each hexagon represents 50 to 100 people"

    def test_scale_text_collapses_when_values_equal(self):
        text = HexMapVisual._scale_text(100, 100)
        assert text == "Each hexagon represents ~100 people"


class TestHexTextFit:
    RADIUS = 1.0
    DX = math.sqrt(3) * RADIUS
    DY = 1.5 * RADIUS

    def _fit(self, points):
        return HexTextFit.best_label_fit(points, self.RADIUS)

    def test_horizontal_row_is_zero_degrees(self):
        points = [(col * self.DX, 0.0) for col in range(4)]
        _, _, width, _, angle = self._fit(points)
        assert angle == 0.0
        assert width == 4 * self.DX

    def test_up_right_chain_is_sixty_degrees(self):
        points = [(k * self.DX / 2, k * self.DY) for k in range(3)]
        _, _, _, _, angle = self._fit(points)
        assert angle == 60.0

    def test_down_right_chain_is_minus_sixty_degrees(self):
        points = [(k * self.DX / 2, -k * self.DY) for k in range(3)]
        _, _, _, _, angle = self._fit(points)
        assert angle == -60.0

    def test_single_hex_defaults_to_horizontal_at_its_center(self):
        cx, cy, _, _, angle = self._fit([(5.0, 7.0)])
        assert (cx, cy) == (5.0, 7.0)
        assert angle == 0.0

    def test_longest_sequence_wins_over_shorter_ones(self):
        row = [(col * self.DX, 0.0) for col in range(5)]
        diagonal = [(k * self.DX / 2, k * self.DY) for k in range(3)]
        _, _, _, _, angle = self._fit(row + diagonal)
        assert angle == 0.0

    def test_center_is_midpoint_of_the_sequence(self):
        points = [(col * self.DX, 0.0) for col in range(4)]
        cx, cy, _, _, _ = self._fit(points)
        assert cx == 1.5 * self.DX
        assert cy == 0.0
