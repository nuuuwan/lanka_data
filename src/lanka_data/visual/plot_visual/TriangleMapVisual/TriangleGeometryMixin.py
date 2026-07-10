import math


class TriangleGeometryMixin:
    @staticmethod
    def _row_height(size):
        return size * math.sqrt(3) / 2

    @classmethod
    def _is_up(cls, y, origin_y, size):
        height = cls._row_height(size)
        frac = ((y - origin_y) % height) / height
        return frac < 0.5

    @classmethod
    def _vertices(cls, x, y, size, origin_y):
        height = cls._row_height(size)
        half = size / 2
        if cls._is_up(y, origin_y, size):
            base = y - height / 3
            return [
                (x - half, base),
                (x + half, base),
                (x, base + height),
            ]
        top = y + height / 3
        return [
            (x - half, top),
            (x + half, top),
            (x, top - height),
        ]
