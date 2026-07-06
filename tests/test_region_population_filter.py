from lanka_data.visual.plot.map import RegionPopulationFilter


class TestRegionPopulationFilter:
    def test_zero_population_regions_are_removed(self):
        data_list = [
            {"region_id": "A", "total_value": 0},
            {"region_id": "B", "total_value": 100},
            {"region_id": "C", "total_value": 200},
        ]
        kept = {
            d["region_id"] for d in RegionPopulationFilter.filter(data_list)
        }
        assert kept == {"B", "C"}

    def test_none_population_regions_are_removed(self):
        data_list = [
            {"region_id": "A", "total_value": None},
            {"region_id": "B", "total_value": 100},
        ]
        kept = {
            d["region_id"] for d in RegionPopulationFilter.filter(data_list)
        }
        assert kept == {"B"}

    def test_tiny_region_below_one_percent_is_removed(self):
        data_list = [
            {"region_id": "A", "total_value": 0.5},
            {"region_id": "B", "total_value": 100},
            {"region_id": "C", "total_value": 200},
        ]
        kept = {
            d["region_id"] for d in RegionPopulationFilter.filter(data_list)
        }
        assert kept == {"B", "C"}

    def test_region_at_or_above_one_percent_is_kept(self):
        data_list = [
            {"region_id": "A", "total_value": 2},
            {"region_id": "B", "total_value": 100},
        ]
        kept = {
            d["region_id"] for d in RegionPopulationFilter.filter(data_list)
        }
        assert kept == {"A", "B"}

    def test_original_order_is_preserved(self):
        data_list = [
            {"region_id": "C", "total_value": 200},
            {"region_id": "A", "total_value": 0},
            {"region_id": "B", "total_value": 100},
        ]
        ids = [
            d["region_id"] for d in RegionPopulationFilter.filter(data_list)
        ]
        assert ids == ["C", "B"]
