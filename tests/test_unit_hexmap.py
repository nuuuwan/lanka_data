from lanka_data.api.fields.How import How
from lanka_data.visual.plot.map.HexData import HexData
from lanka_data.visual.plot.map.HexData.UnitHexData import UnitHexData
from lanka_data.visual.plot_visual.HexMapVisual import HexMapVisual
from lanka_data.visual.plot_visual.UnitHexMapVisual import UnitHexMapVisual
from lanka_data.visual.VisualFactory import VisualFactory


class TestUnitHexDataCounts:
    def test_every_region_gets_exactly_one_hex(self):
        counts = UnitHexData.get_counts({"A": 900, "B": 100, "C": 1})
        assert counts == {"A": 1, "B": 1, "C": 1}

    def test_counts_ignore_weight_magnitude(self):
        counts = UnitHexData.get_counts({"A": 0, "B": 1_000_000})
        assert set(counts.values()) == {1}


class TestUnitHexDataCache:
    def test_cache_prefix_differs_from_hexmap(self):
        weights = {"A": 900, "B": 100}
        assert HexData._cache_path(weights) != UnitHexData._cache_path(
            weights
        )
        assert "unit_hex_" in UnitHexData._cache_path(weights)


class TestUnitHexMapRouting:
    def test_unit_hexmap_base_is_registered(self):
        assert "UnitHexMap" in How.BASE_LABELS
        assert "UnitHexMap" in How.CATEGORY_BASES

    def test_factory_maps_unit_hexmap_to_unit_hexmap_visual(self):
        assert VisualFactory._VISUAL_CLS["UnitHexMap"] is UnitHexMapVisual

    def test_unit_hexmap_visual_subclasses_hexmap_visual(self):
        assert issubclass(UnitHexMapVisual, HexMapVisual)


class TestUnitHexMapLayout:
    def test_layout_has_one_hex_per_region(self):
        data_list = [
            {"region_id": "LK-1", "region_name": "LK-1", "total_value": 900},
            {"region_id": "LK-2", "region_name": "LK-2", "total_value": 100},
            {"region_id": "LK-3", "region_name": "LK-3", "total_value": 1},
        ]
        layout = UnitHexMapVisual._get_layout(data_list)
        ids = sorted(region_id for region_id, _, _ in layout["hexes"])
        assert ids == ["LK-1", "LK-2", "LK-3"]

    def test_get_data_list_keeps_all_regions(self):
        data_list = [
            {"region_id": "A", "total_value": 1000},
            {"region_id": "B", "total_value": 1},
        ]

        class FakeDataset:
            def get_data_table(self):
                return data_list

        assert UnitHexMapVisual._get_data_list(FakeDataset()) == data_list

    def test_draw_scale_is_suppressed(self):
        assert UnitHexMapVisual._draw_scale(None, {}) is None
