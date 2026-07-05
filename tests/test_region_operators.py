from lanka_data.region.RegionParserMixin.RegionParserMixin import \
    RegionParserMixin
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

    def test_radius_operator_expands_by_radius(self, monkeypatch):
        self.setup_region_year_mock(monkeypatch)
        monkeypatch.setattr(
            RegionParserMixin,
            "get_region_ids_from_region_radius",
            classmethod(
                lambda cls, region_id, radius_km: [
                    "LK-1127025",
                    "LK-1127030",
                ]
            ),
        )
        parent_ids, region_year = RegionParserMixin.parse_parent_part(
            "LK-1127025@10"
        )
        assert parent_ids == ["LK-1127025", "LK-1127030"]
        assert region_year == "Current"

    def test_radius_formatter_names_radius_not_zoom(self, monkeypatch):
        monkeypatch.setattr(
            WhereFormatter,
            "format_regions",
            lambda self, ids: "Region " + ids[0],
        )
        assert (
            WhereFormatter("LK-1127025@10").format()
            == "Within 10km of Region LK-1127025"
        )
