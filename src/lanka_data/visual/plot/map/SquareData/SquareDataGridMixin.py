import math


class SquareDataGridMixin:
    GRID_FACTOR = 1.3
    MAX_GRID_ITERATIONS = 12

    @staticmethod
    def _square_centers(bounds, size):
        minx, miny, maxx, maxy = bounds
        side = 2 * size
        centers = []
        y = miny
        while y <= maxy + side:
            x = minx
            while x <= maxx + side:
                centers.append((x, y))
                x += side
            y += side
        return centers

    @classmethod
    def _initial_size(cls, bounds, target):
        minx, miny, maxx, maxy = bounds
        area = max((maxx - minx) * (maxy - miny), 1e-12)
        side = math.sqrt(area / max(target, 1))
        return side / 2

    @classmethod
    def build_grid(cls, bounds, total_count):
        target = max(total_count * cls.GRID_FACTOR, total_count + 1)
        size = cls._initial_size(bounds, target)
        centers = cls._square_centers(bounds, size)
        for _ in range(cls.MAX_GRID_ITERATIONS):
            if len(centers) >= total_count:
                break
            size *= 0.85
            centers = cls._square_centers(bounds, size)
        return centers, size
