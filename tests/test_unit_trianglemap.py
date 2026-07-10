from lanka_data.api.fields.How import How
from lanka_data.visual.plot.map.TriangleData.TriangleData import TriangleData
from lanka_data.visual.plot.map.TriangleData.UnitTriangleData import (
    UnitTriangleData,
)
from lanka_data.visual.plot_visual.TriangleMapVisual.TriangleMapVisual import (
    TriangleMapVisual,
)
from lanka_data.visual.plot_visual.UnitTriangleMapVisual.UnitTriangleMapVisual import (
    UnitTriangleMapVisual,
)
from lanka_data.visual.VisualFactory import VisualFactory


class TestUnitTriangleDataCounts:
    def test_every_region_gets_exactly_one_triangle(self):
        counts = UnitTriangleData.get_counts({"A": 900, "B": 100, "C": 1})
        assert counts == {"A": 1, "B": 1, "C": 1}

    def test_counts_ignore_weight_magnitude(self):
        counts = UnitTriangleData.get_counts({"A": 0, "B": 1_000_000})
        assert set(counts.values()) == {1}


class TestUnitTriangleDataCache:
    def test_cache_prefix_differs_from_trianglemap(self):
        weights = {"A": 900, "B": 100}
        assert TriangleData._cache_path(weights) != (
            UnitTriangleData._cache_path(weights)
        )
        assert "unit_triangle_" in UnitTriangleData._cache_path(weights)


class TestUnitTriangleMapRouting:
    def test_unit_trianglemap_base_is_registered(self):
        assert "UnitTriangleMap" in How.BASE_LABELS
        assert "UnitTriangleMap" in How.CATEGORY_BASES

    def test_factory_maps_unit_trianglemap_to_visual(self):
        assert (
            VisualFactory._VISUAL_CLS["UnitTriangleMap"]
            is UnitTriangleMapVisual
        )

    def test_unit_visual_subclasses_trianglemap_visual(self):
        assert issubclass(UnitTriangleMapVisual, TriangleMapVisual)


class TestUnitTriangleMapLayout:
    DATA_A = [
        {"region_id": "LK-1", "region_name": "LK-1", "total_value": 900},
        {"region_id": "LK-2", "region_name": "LK-2", "total_value": 100},
        {"region_id": "LK-3", "region_name": "LK-3", "total_value": 1},
    ]
    DATA_B = [
        {"region_id": "LK-1", "region_name": "LK-1", "total_value": 1},
        {"region_id": "LK-2", "region_name": "LK-2", "total_value": 500},
        {"region_id": "LK-3", "region_name": "LK-3", "total_value": 900},
    ]

    def test_layout_has_one_triangle_per_region(self):
        layout = UnitTriangleMapVisual._get_layout(self.DATA_A)
        ids = sorted(region_id for region_id, _, _ in layout["triangles"])
        assert ids == ["LK-1", "LK-2", "LK-3"]

    def test_layout_is_same_across_differing_weights(self):
        layout_a = UnitTriangleMapVisual._get_layout(self.DATA_A)
        layout_b = UnitTriangleMapVisual._get_layout(self.DATA_B)
        assert layout_a["triangles"] == layout_b["triangles"]

    def test_get_data_list_keeps_all_regions(self):
        data_list = [
            {"region_id": "A", "total_value": 1000},
            {"region_id": "B", "total_value": 1},
        ]

        class FakeDataset:
            def get_data_table(self):
                return data_list

        assert (
            UnitTriangleMapVisual._get_data_list(FakeDataset()) == data_list
        )

    def test_draw_scale_is_suppressed(self):
        assert UnitTriangleMapVisual._draw_scale(None, {}) is None
