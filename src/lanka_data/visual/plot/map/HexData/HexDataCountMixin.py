from utils_future import Log

log = Log("HexData")


class HexDataCountMixin:
    HEXMAP_ERROR = 0.25

    @staticmethod
    def _region_error(actual, ideal):
        return abs(actual - ideal) / ideal

    @classmethod
    def _max_error(cls, weights, value_per_hex):
        errors = []
        for weight in weights:
            ideal = weight / value_per_hex
            actual = max(1, round(ideal))
            errors.append(cls._region_error(actual, ideal))
        return max(errors)

    @classmethod
    def _candidates(cls, weights, cap):
        n_max = int(0.5 / cls.HEXMAP_ERROR) + 2
        values = {min(weights) * 2 * cls.HEXMAP_ERROR}
        for weight in weights:
            for n in range(1, n_max + 1):
                value = weight * (1 + cls.HEXMAP_ERROR) / n
                if value <= cap:
                    values.add(value)
        return sorted(values, reverse=True)

    @classmethod
    def _value_per_hex(cls, region_to_weight):
        weights = [w for w in region_to_weight.values() if w > 0]
        if not weights:
            return None
        cap = min(weights) * (1 + cls.HEXMAP_ERROR)
        tolerance = cls.HEXMAP_ERROR + 1e-9
        for value in cls._candidates(weights, cap):
            if cls._max_error(weights, value) <= tolerance:
                return value
        return min(weights) * 2 * cls.HEXMAP_ERROR

    @classmethod
    def _log_region(cls, region_id, actual, ideal):
        if ideal <= 0:
            return
        error = cls._region_error(actual, ideal)
        emoji = "" if error <= cls.HEXMAP_ERROR else "⚠️"
        log.debug(
            f"{region_id}: actual={actual} "
            + f"ideal={ideal:.2f} error={error:.0%} {emoji}"
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
