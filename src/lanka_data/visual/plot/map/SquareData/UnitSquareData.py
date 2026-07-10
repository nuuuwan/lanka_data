from lanka_data.visual.plot.map.SquareData.SquareData import SquareData


class UnitSquareData(SquareData):
    CACHE_PREFIX = "unit_square"

    @classmethod
    def get_counts(cls, region_to_weight):
        return {region_id: 1 for region_id in region_to_weight}
