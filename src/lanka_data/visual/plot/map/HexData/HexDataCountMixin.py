from utils_future import Log

log = Log("HexData")


class HexDataCountMixin:
    HEXMAP_ERROR = 0.5

    @classmethod
    def _value_per_hex(cls, region_to_weight):
        weights = [w for w in region_to_weight.values() if w > 0]
        if not weights:
            return None
        return min(weights) * (1 + cls.HEXMAP_ERROR)

    @staticmethod
    def _region_error(actual, ideal):
        return abs(actual - ideal)

    @classmethod
    def _log_region(cls, region_id, actual, ideal):
        error = cls._region_error(actual, ideal)
        log.debug(
            f"{region_id}: actual={actual} "
            + f"ideal={ideal:.2f} error={error:.2f}"
        )

    @classmethod
    def get_counts(cls, region_to_weight):
        value_per_hex = cls._value_per_hex(region_to_weight)
        if value_per_hex is None:
            return {region_id: 1 for region_id in region_to_weight}
        counts = {}
        for region_id, weight in region_to_weight.items():
            ideal = weight / value_per_hex
            actual = max(1, round(ideal))
            counts[region_id] = actual
            cls._log_region(region_id, actual, ideal)
        return counts
