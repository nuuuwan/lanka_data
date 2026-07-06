import math


class BubbleDataRadiusMixin:
    FILL_RATIO = 0.35
    MIN_RADIUS_RATIO = 0.01

    @staticmethod
    def _bounds_area(bounds):
        minx, miny, maxx, maxy = bounds
        return max((maxx - minx) * (maxy - miny), 1e-12)

    @staticmethod
    def _min_radius(bounds):
        minx, miny, maxx, maxy = bounds
        span = max(maxx - minx, maxy - miny)
        return span * BubbleDataRadiusMixin.MIN_RADIUS_RATIO

    @classmethod
    def get_radii(cls, region_to_weight, bounds):
        total_weight = sum(region_to_weight.values()) or 1
        scale = cls.FILL_RATIO * cls._bounds_area(bounds) / total_weight
        min_radius = cls._min_radius(bounds)
        return {
            region_id: max(math.sqrt(scale * weight / math.pi), min_radius)
            for region_id, weight in region_to_weight.items()
        }
