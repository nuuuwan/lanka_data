import math


class TriangleDataGridMixin:
    GRID_FACTOR = 1.3
    MAX_GRID_ITERATIONS = 12
    TRIANGLE_AREA_FACTOR = math.sqrt(3) / 4

    @staticmethod
    def _triangle_centers(bounds, size):
        minx, miny, maxx, maxy = bounds
        height = size * math.sqrt(3) / 2
        centers = []
        y0 = miny
        while y0 <= maxy + height:
            k = 0
            while minx + k * size <= maxx + size:
                base = minx + k * size
                centers.append((base + size / 2, y0 + height / 3, True))
                centers.append((base + size, y0 + 2 * height / 3, False))
                k += 1
            y0 += height
        return centers

    @classmethod
    def _initial_size(cls, bounds, target):
        minx, miny, maxx, maxy = bounds
        area = max((maxx - minx) * (maxy - miny), 1e-12)
        return math.sqrt(area / (max(target, 1) * cls.TRIANGLE_AREA_FACTOR))

    @classmethod
    def build_grid(cls, bounds, total_count):
        target = max(total_count * cls.GRID_FACTOR, total_count + 1)
        size = cls._initial_size(bounds, target)
        centers = cls._triangle_centers(bounds, size)
        for _ in range(cls.MAX_GRID_ITERATIONS):
            if len(centers) >= total_count:
                break
            size *= 0.85
            centers = cls._triangle_centers(bounds, size)
        return centers, size
