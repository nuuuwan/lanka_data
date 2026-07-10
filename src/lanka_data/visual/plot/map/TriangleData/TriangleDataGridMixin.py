import math


class TriangleDataGridMixin:
    GRID_FACTOR = 1.3
    MAX_GRID_ITERATIONS = 12

    @staticmethod
    def _row_height(size):
        return size * math.sqrt(3) / 2

    @classmethod
    def _row_centers(cls, bounds, size, row_index, y_base):
        minx, _, maxx, _ = bounds
        height = cls._row_height(size)
        centers = []
        column = 0
        while minx + column * size / 2 + size / 2 <= maxx + size:
            cx = minx + column * size / 2 + size / 2
            up = (column + row_index) % 2 == 0
            cy = y_base + (height / 3 if up else 2 * height / 3)
            centers.append((cx, cy))
            column += 1
        return centers

    @classmethod
    def _triangle_centers(cls, bounds, size):
        _, miny, _, maxy = bounds
        height = cls._row_height(size)
        centers = []
        row_index = 0
        y_base = miny
        while y_base <= maxy + height:
            centers.extend(cls._row_centers(bounds, size, row_index, y_base))
            y_base += height
            row_index += 1
        return centers

    @classmethod
    def _initial_size(cls, bounds, target):
        minx, miny, maxx, maxy = bounds
        area = max((maxx - minx) * (maxy - miny), 1e-12)
        return math.sqrt(4 * area / (max(target, 1) * math.sqrt(3)))

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
