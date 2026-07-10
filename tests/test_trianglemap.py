from lanka_data.visual.plot.map.TriangleData.TriangleData import TriangleData
from lanka_data.visual.plot_visual.TriangleMapVisual.TriangleMapVisual import (
    TriangleMapVisual,
)
from lanka_data.visual.plot_visual.TriangleMapVisual.TriangleMapBoundaryMixin import (
    TriangleMapBoundaryMixin,
)
from lanka_data.visual.plot_visual.TriangleMapVisual.TriangleGeometryMixin import (
    TriangleGeometryMixin,
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

    def test_nearest_neighbours_are_uniform(self):
        centers, _ = TriangleData.build_grid((0, 0, 10, 10), 20)
        adj = TriangleData._adjacency(centers)
        assert all(len(neighbours) <= 3 for neighbours in adj.values())
        assert any(len(neighbours) == 3 for neighbours in adj.values())

    def test_assign_gives_each_slot_a_distinct_cell(self):
        centroids = {"A": (0, 0), "B": (10, 10)}
        counts = {"A": 2, "B": 1}
        centers, _ = TriangleData.build_grid((0, 0, 10, 10), 3)
        triangles = TriangleData.assign(centroids, counts, centers)
        assert len(triangles) == 3
        used = {(x, y) for _, x, y in triangles}
        assert len(used) == 3
        region_ids = [region_id for region_id, _, _ in triangles]
        assert region_ids.count("A") == 2
        assert region_ids.count("B") == 1


class TestTriangleMapRouting:
    def test_trianglemap_base_is_registered(self):
        from lanka_data.api.fields.How import How

        assert "TriangleMap" in How.BASE_LABELS
        assert "TriangleMap" in How.CATEGORY_BASES

    def test_factory_maps_trianglemap_to_visual(self):
        assert VisualFactory._VISUAL_CLS["TriangleMap"] is TriangleMapVisual


class TestTriangleScaleRange:
    def test_range_reflects_lowest_and_highest_per_region(self):
        region_to_weight = {"A": 100, "B": 100}
        triangles = [("A", 0, 0), ("A", 1, 0), ("B", 2, 0)]
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


class TestTriangleGeometry:
    SIZE = 2.0

    def test_centroid_of_vertices_matches_center(self):
        centers, size = TriangleData.build_grid((0, 0, 10, 10), 20)
        for x, y in centers[:10]:
            vertices = TriangleGeometryMixin._vertices(x, y, size, 0.0)
            cx = sum(p[0] for p in vertices) / 3
            cy = sum(p[1] for p in vertices) / 3
            assert abs(cx - x) < 1e-9
            assert abs(cy - y) < 1e-9

    def test_rows_alternate_up_and_down(self):
        centers, size = TriangleData.build_grid((0, 0, 10, 10), 20)
        orientations = {
            TriangleGeometryMixin._is_up(y, 0.0, size) for _, y in centers
        }
        assert orientations == {True, False}


class TestTriangleBoundary:
    @staticmethod
    def _layout():
        centers, size = TriangleData.build_grid((0, 0, 10, 10), 30)
        return {
            "size": size,
            "origin_y": 0.0,
            "triangles": [["A", x, y] for x, y in centers],
        }

    def test_contiguous_region_merges_into_single_polygon(self):
        layout = self._layout()
        polys = TriangleMapBoundaryMixin._region_to_polygons(layout)["A"]
        merged = TriangleMapBoundaryMixin._merge(polys, layout["size"])
        assert merged.geom_type == "Polygon"

    def test_draw_boundaries_plots_at_least_one_ring(self):
        layout = self._layout()
        drawn = []

        class FakeAx:
            def plot(self, xs, ys, **kwargs):
                drawn.append((list(xs), list(ys)))

        TriangleMapBoundaryMixin._draw_boundaries(FakeAx(), layout)
        assert len(drawn) >= 1


class TestTriangleContiguity:
    @classmethod
    def _components_of(cls, repaired, centers):
        adj = TriangleData._adjacency(centers)
        pos = {
            (round(x, 9), round(y, 9)): i for i, (x, y) in enumerate(centers)
        }
        by_region = {}
        for r, x, y in repaired:
            idx = pos[(round(x, 9), round(y, 9))]
            by_region.setdefault(r, []).append(idx)
        return {
            r: TriangleData._components(idxs, adj)
            for r, idxs in by_region.items()
        }

    def test_orphan_cell_is_reconnected(self):
        centers, _ = TriangleData.build_grid((0, 0, 10, 10), 30)
        first = list(centers[0])
        far = list(centers[-1])
        cells = [["A", first[0], first[1]], ["A", far[0], far[1]]]
        repaired = TriangleData.repair(cells, centers)
        ids = [r for r, _, _ in repaired]
        assert ids.count("A") == 2
        comps = self._components_of(repaired, centers)
        assert len(comps["A"]) == 1
