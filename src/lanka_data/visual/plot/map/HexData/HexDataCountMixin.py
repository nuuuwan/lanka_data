class HexDataCountMixin:
    TARGET_TOTAL_HEXES = 100

    @classmethod
    def get_counts(cls, region_to_weight):
        total = sum(region_to_weight.values()) or 1
        scale = total / cls.TARGET_TOTAL_HEXES
        counts = {}
        for region_id, weight in region_to_weight.items():
            counts[region_id] = max(1, round(weight / scale))
        return counts
