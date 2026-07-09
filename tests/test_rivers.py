from lanka_data.datasets.dataset.custom.RiversDataset import RiversDataset
from lanka_data.datasets.region.RegionParserMixin import RegionParserMixin
from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils
from lanka_data.datasets.region.rivers.RiversData import RiversData

SAMPLE_FEATURES = [
    {
        "properties": {
            "MAIN_RIV": 111,
            "LENGTH_KM": 1.0,
            "UPLAND_SKM": 5.0,
            "CATCH_SKM": 2.0,
        },
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [[[80.0, 9.0], [80.1, 9.1]]],
        },
    },
    {
        "properties": {
            "MAIN_RIV": 111,
            "LENGTH_KM": 2.0,
            "UPLAND_SKM": 3.0,
            "CATCH_SKM": 3.0,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [[80.1, 9.1], [80.2, 9.2]],
        },
    },
    {
        "properties": {
            "MAIN_RIV": 222,
            "LENGTH_KM": 4.0,
            "UPLAND_SKM": 8.0,
            "CATCH_SKM": 1.0,
        },
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
        assert region["region_id"] == "R-111"
        assert region["region_name"] == "River 111"
        assert region["region_type"] == "rivers"
        assert region["geometry"]["type"] == "MultiLineString"

    def test_build_region_has_center(self):
        region = RiversData._build_region(111, [[[80.0, 9.0], [80.2, 9.4]]])
        assert region["center_lat"] == 9.2
        assert region["center_lng"] == 80.1

    def test_aggregate_measures(self):
        measures = {}
        for feature in SAMPLE_FEATURES:
            RiversData._accumulate_measures(measures, feature)
        assert measures["R-111"]["RiverLen"] == 3.0
        assert measures["R-111"]["Catchment"] == 5.0
        assert measures["R-222"]["RiverLen"] == 4.0
        assert measures["R-222"]["Catchment"] == 8.0


class TestRiversDataset:
    def test_labels(self):
        assert RiversDataset.get_labels() == ["RiverLen", "Catchment"]

    def test_supports(self):
        assert RiversDataset.supports("RiverLen", "2026")
        assert RiversDataset.supports("Catchment", "2026")
        assert not RiversDataset.supports("RiverLen", "2024")
        assert not RiversDataset.supports("Population", "2026")

    def test_source_data_table(self, monkeypatch):
        monkeypatch.setattr(
            RiversData,
            "get_river_measures",
            classmethod(
                lambda cls: {"R-111": {"RiverLen": 3.0, "Catchment": 5.0}}
            ),
        )
        dataset = RiversDataset([{"region_id": "R-111"}], "RiverLen")
        rows = dataset.get_source_data_table()
        assert rows == [{"region_id": "R-111", "values": {"RiverLen": 3.0}}]


class TestRiversRegionType:
    def test_get_region_type_for_river_id(self):
        assert RegionTypeUtils.get_region_type("R-111") == "rivers"

    def test_long_name_plural_for_rivers(self):
        assert RegionTypeUtils.get_long_name_plural("rivers") == "Rivers"


class TestRiversParsing:
    def test_get_raw_regions_routes_to_rivers(self, monkeypatch):
        sentinel = [{"region_id": "R-111"}]
        monkeypatch.setattr(
            RiversData, "get_river_regions", staticmethod(lambda: sentinel)
        )
        regions, region_year = RegionParserMixin.get_raw_regions(
            ["LK"], "rivers", "Current"
        )
        assert regions is sentinel
        assert region_year == "Current"

    def test_get_raw_regions_routes_single_river_id(self, monkeypatch):
        river_regions = [
            {"region_id": "R-111"},
            {"region_id": "R-222"},
        ]
        monkeypatch.setattr(
            RiversData,
            "get_river_regions",
            staticmethod(lambda: river_regions),
        )
        regions, region_year = RegionParserMixin.get_raw_regions(
            ["R-222"], None, "Current"
        )
        assert regions == [{"region_id": "R-222"}]
        assert region_year == "Current"
