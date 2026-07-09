from lanka_data.datasets.region.RegionParserMixin import RegionParserMixin
from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils
from lanka_data.datasets.region.rivers.RiversData import RiversData

SAMPLE_FEATURES = [
    {
        "properties": {"MAIN_RIV": 111},
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [[[80.0, 9.0], [80.1, 9.1]]],
        },
    },
    {
        "properties": {"MAIN_RIV": 111},
        "geometry": {
            "type": "LineString",
            "coordinates": [[80.1, 9.1], [80.2, 9.2]],
        },
    },
    {
        "properties": {"MAIN_RIV": 222},
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [[[81.0, 8.0], [81.1, 8.1]]],
        },
    },
]


class TestRiversData:
    def test_group_by_main_river_merges_segments(self):
        grouped = RiversData._group_by_main_river(SAMPLE_FEATURES)
        assert set(grouped.keys()) == {111, 222}
        assert len(grouped[111]) == 2
        assert len(grouped[222]) == 1

    def test_build_region_shape(self):
        region = RiversData._build_region(111, [[[80.0, 9.0], [80.1, 9.1]]])
        assert region["region_id"] == "LK-river-111"
        assert region["region_name"] == "River 111"
        assert region["region_type"] == "rivers"
        assert region["geometry"]["type"] == "MultiLineString"


class TestRiversRegionType:
    def test_get_region_type_for_river_id(self):
        assert RegionTypeUtils.get_region_type("LK-river-111") == "rivers"

    def test_long_name_plural_for_rivers(self):
        assert RegionTypeUtils.get_long_name_plural("rivers") == "Rivers"


class TestRiversParsing:
    def test_get_raw_regions_routes_to_rivers(self, monkeypatch):
        sentinel = [{"region_id": "LK-river-111"}]
        monkeypatch.setattr(
            RiversData, "get_river_regions", staticmethod(lambda: sentinel)
        )
        regions, region_year = RegionParserMixin.get_raw_regions(
            ["LK"], "rivers", "Current"
        )
        assert regions is sentinel
        assert region_year == "Current"
