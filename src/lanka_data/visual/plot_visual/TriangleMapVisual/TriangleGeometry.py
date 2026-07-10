import math


class TriangleGeometry:
    @staticmethod
    def vertices(cx, cy, size, up):
        height = size * math.sqrt(3) / 2
        if up:
            return [
                (cx - size / 2, cy - height / 3),
                (cx + size / 2, cy - height / 3),
                (cx, cy + 2 * height / 3),
            ]
        return [
            (cx - size / 2, cy + height / 3),
            (cx + size / 2, cy + height / 3),
            (cx, cy - 2 * height / 3),
        ]
