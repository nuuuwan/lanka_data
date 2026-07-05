class HexDataCountMixin:
    HEXMAP_ERROR = 0.5

    @classmethod
    def _value_per_hex(cls, region_to_weight):
        weights = [w for w in region_to_weight.values() if w > 0]
        if not weights:
            return None
        return min(weights) / cls.HEXMAP_ERROR

    @classmethod
    def get_counts(cls, region_to_weight):
        value_per_hex = cls._value_per_hex(region_to_weight)
        if value_per_hex is None:
            return {region_id: 1 for region_id in region_to_weight}
        counts = {}
        for region_id, weight in region_to_weight.items():
            counts[region_id] = max(1, round(weight / value_per_hex))
        return counts
