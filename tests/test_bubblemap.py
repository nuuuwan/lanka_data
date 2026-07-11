import math

from lanka_data.visual.plot.map.BubbleData.BubbleData import BubbleData
from lanka_data.visual.plot_visual.BubbleMapVisual.BubbleMapVisual import \
    BubbleMapVisual
from lanka_data.visual.VisualFactory import VisualFactory


class TestBubbleDataRadius:
    BOUNDS = (0, 0, 100, 100)

    def test_radius_is_population_proportional(self):
        radii = BubbleData.get_radii({"A": 900, "B": 100}, self.BOUNDS)
        assert radii["A"] > radii["B"]

    def test_area_ratio_matches_population_ratio(self):
        radii = BubbleData.get_radii({"A": 400, "B": 100}, self.BOUNDS)
        area_ratio = (radii["A"] ** 2) / (radii["B"] ** 2)
        assert abs(area_ratio - 4.0) < 1e-6

    def test_tiny_region_gets_min_radius(self):
        radii = BubbleData.get_radii({"A": 1_000_000, "B": 1}, self.BOUNDS)
        assert radii["B"] >= BubbleData._min_radius(self.BOUNDS) - 1e-9


class TestBubbleDataRelax:
    BOUNDS = (0, 0, 100, 100)

    @staticmethod
    def _min_gap(bubbles):
        gap = float("inf")
        for i in range(len(bubbles)):
            for j in range(i + 1, len(bubbles)):
                _, x1, y1, r1 = bubbles[i]
                _, x2, y2, r2 = bubbles[j]
                gap = min(gap, math.hypot(x2 - x1, y2 - y1) - (r1 + r2))
        return gap

    def test_overlapping_bubbles_are_separated(self):
        radii = {"A": 20, "B": 20, "C": 20}
        centroids = {"A": (50, 50), "B": (51, 51), "C": (49, 49)}
        bubbles = BubbleData.relax(centroids, radii, self.BOUNDS)
        assert self._min_gap(bubbles) >= -1e-6

    def test_relax_keeps_every_region(self):
        radii = {"A": 5, "B": 5}
        centroids = {"A": (10, 10), "B": (90, 90)}
        bubbles = BubbleData.relax(centroids, radii, self.BOUNDS)
        assert {b[0] for b in bubbles} == {"A", "B"}

    def test_coincident_centroids_are_pushed_apart(self):
        radii = {"A": 10, "B": 10}
        centroids = {"A": (50, 50), "B": (50, 50)}
        bubbles = BubbleData.relax(centroids, radii, self.BOUNDS)
        assert self._min_gap(bubbles) >= -1e-6

    def test_bubbles_stay_within_bounds(self):
        radii = {"A": 20, "B": 20, "C": 20, "D": 20}
        centroids = {
            "A": (50, 50),
            "B": (51, 50),
            "C": (50, 51),
            "D": (49, 49),
        }
        bubbles = BubbleData.relax(centroids, radii, self.BOUNDS)
        minx, miny, maxx, maxy = self.BOUNDS
        for _, x, y, r in bubbles:
            assert x - r >= minx - 1e-6
            assert x + r <= maxx + 1e-6
            assert y - r >= miny - 1e-6
            assert y + r <= maxy + 1e-6


class TestBubbleMapRouting:
    def test_bubblemap_base_is_registered(self):
        from lanka_data.api.fields.How import How

        assert "BubbleMap" in How.BASE_LABELS

    def test_bubblemap_is_a_category_base(self):
        from lanka_data.api.fields.How import How

        assert "BubbleMap" in How.CATEGORY_BASES

    def test_factory_maps_bubblemap_to_bubblemap_visual(self):
        assert VisualFactory._VISUAL_CLS["BubbleMap"] is BubbleMapVisual


class TestBubbleMapDraw:
    def test_bubble_positions_extracts_centers(self):
        layout = {"bubbles": [("A", 1, 2, 5), ("B", 3, 4, 6)]}
        positions = BubbleMapVisual._bubble_positions(layout)
        assert positions == {"A": (1, 2), "B": (3, 4)}

    def test_bubbles_drawn_as_circles(self):
        layout = {"bubbles": [("A", 0, 0, 5)]}
        patches = []

        class FakeAx:
            def add_patch(self, patch):
                patches.append(patch)

        BubbleMapVisual._draw_bubbles(FakeAx(), layout, {"A": "#8D153A"})
        assert len(patches) == 1
        assert patches[0].radius == 5
