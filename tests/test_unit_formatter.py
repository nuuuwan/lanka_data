from lanka_data.visual.formatters.UnitFormatter import UnitFormatter


class TestUnitFormatter:
    def test_river_len_unit_is_km(self):
        assert UnitFormatter("RiverLen").unit() == "km"

    def test_catchment_unit_is_sqkm(self):
        assert UnitFormatter("Catchment").unit() == "sqkm"

    def test_unknown_label_has_no_unit(self):
        assert UnitFormatter("Religion").unit() is None

    def test_format_appends_unit(self):
        assert UnitFormatter("RiverLen").format("330") == "330 km"

    def test_format_without_unit_is_unchanged(self):
        assert UnitFormatter("Religion").format("5") == "5"
