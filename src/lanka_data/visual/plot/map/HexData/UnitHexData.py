from lanka_data.visual.plot.map.HexData.HexData import HexData


class UnitHexData(HexData):
    CACHE_PREFIX = "unit_hex"

    @classmethod
    def _prepare_data_list(cls, data_list):
        return [{**d, "total_value": 1} for d in data_list]

    @classmethod
    def get_counts(cls, region_to_weight):
        return {region_id: 1 for region_id in region_to_weight}
