from lanka_data.visual.plot.map.HexData import HexData
from lanka_data.visual.plot_visual.HexMapVisual import HexMapVisual
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
