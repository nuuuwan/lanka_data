from lanka_data.region.RegionParserMixin.RegionParserMixin import (
    RegionParserMixin,
)
from lanka_data.visual.formatters.WhereFormatter import WhereFormatter


class TestRegionOperators:
    def setup_region_year_mock(self, monkeypatch):
        monkeypatch.setattr(
            RegionParserMixin,
            "_get_region_year",
            classmethod(
                lambda cls, x: (
                    x.split("-pre")[1] if "-pre" in x else "Current"
                )
            ),
            raising=False,
        )

    def test_range_parent_operator(self, monkeypatch):
        self.setup_region_year_mock(monkeypatch)
        monkeypatch.setattr(
            RegionParserMixin,
            "get_region_ids_from_range",
            classmethod(lambda cls, a, b: ["LK-5", "LK-6"]),
        )
        parent_ids, region_year = RegionParserMixin.parse_parent_part(
            "LK-5...LK-6"
        )
        assert parent_ids == ["LK-5", "LK-6"]
        assert region_year == "Current"

    def test_comma_parent_operator(self, monkeypatch):
        self.setup_region_year_mock(monkeypatch)
        parent_ids, region_year = RegionParserMixin.parse_parent_part(
            "LK-1,LK-2"
        )
        assert parent_ids == ["LK-1", "LK-2"]
        assert region_year == "Current"

    def test_pre_year_parent_operator(self, monkeypatch):
        self.setup_region_year_mock(monkeypatch)
        parent_ids, region_year = RegionParserMixin.parse_parent_part(
            "LK-pre1959"
        )
        assert parent_ids == ["LK"]
        assert region_year == "1959"

    def test_zoom_operator_does_not_expand_by_radius(self, monkeypatch):
        self.setup_region_year_mock(monkeypatch)
        parent_ids, region_year = RegionParserMixin.parse_parent_part(
            "LK-1127025@20"
        )
        assert parent_ids == ["LK-1127025"]
        assert region_year == "Current"

    def test_zoom_formatter_names_zoom_not_radius(self, monkeypatch):
        monkeypatch.setattr(
            WhereFormatter,
            "format_regions",
            lambda self, ids: "Region " + ids[0],
        )
        assert (
            WhereFormatter("LK-1127025@20").format()
            == "Region LK-1127025 at zoom 20"
        )
