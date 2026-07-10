from lanka_data.api.fields.How import How
from lanka_data.visual.plot.map.SquareData.SquareData import SquareData
from lanka_data.visual.plot.map.SquareData.UnitSquareData import (
    UnitSquareData,
)
from lanka_data.visual.plot_visual.SquareMapVisual.SquareMapVisual import (
    SquareMapVisual,
)
from lanka_data.visual.plot_visual.UnitSquareMapVisual.UnitSquareMapVisual \
    import UnitSquareMapVisual
from lanka_data.visual.VisualFactory import VisualFactory


class TestUnitSquareDataCounts:
    def test_every_region_gets_exactly_one_square(self):
        counts = UnitSquareData.get_counts({"A": 900, "B": 100, "C": 1})
        assert counts == {"A": 1, "B": 1, "C": 1}

    def test_counts_ignore_weight_magnitude(self):
        counts = UnitSquareData.get_counts({"A": 0, "B": 1_000_000})
        assert set(counts.values()) == {1}


class TestUnitSquareDataCache:
    def test_cache_prefix_differs_from_squaremap(self):
        weights = {"A": 900, "B": 100}
        assert SquareData._cache_path(weights) != (
            UnitSquareData._cache_path(weights)
        )
        assert "unit_square_" in UnitSquareData._cache_path(weights)


class TestUnitSquareMapRouting:
    def test_unit_squaremap_base_is_registered(self):
        assert "UnitSquareMap" in How.BASE_LABELS
        assert "UnitSquareMap" in How.CATEGORY_BASES

    def test_factory_maps_unit_squaremap_to_visual(self):
        assert (
            VisualFactory._VISUAL_CLS["UnitSquareMap"] is UnitSquareMapVisual
        )

    def test_unit_squaremap_visual_subclasses_squaremap_visual(self):
        assert issubclass(UnitSquareMapVisual, SquareMapVisual)


class TestUnitSquareMapLayout:
    def test_layout_has_one_square_per_region(self):
        data_list = [
            {"region_id": "LK-1", "region_name": "LK-1", "total_value": 900},
            {"region_id": "LK-2", "region_name": "LK-2", "total_value": 100},
            {"region_id": "LK-3", "region_name": "LK-3", "total_value": 1},
        ]
        layout = UnitSquareMapVisual._get_layout(data_list)
        ids = sorted(region_id for region_id, _, _ in layout["squares"])
        assert ids == ["LK-1", "LK-2", "LK-3"]

    def test_get_data_list_keeps_all_regions(self):
        data_list = [
            {"region_id": "A", "total_value": 1000},
            {"region_id": "B", "total_value": 1},
        ]

        class FakeDataset:
            def get_data_table(self):
                return data_list

        assert UnitSquareMapVisual._get_data_list(FakeDataset()) == data_list

    def test_draw_scale_is_suppressed(self):
        assert UnitSquareMapVisual._draw_scale(None, {}) is None
