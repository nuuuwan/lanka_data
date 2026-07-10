import math


class TriangleTextFit:
    @staticmethod
    def _bounds(points):
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        return min(xs), min(ys), max(xs), max(ys)

    @classmethod
    def best_label_fit(cls, points, size):
        minx, miny, maxx, maxy = cls._bounds(points)
        height = size * math.sqrt(3) / 2
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2
        rect_w = (maxx - minx) + size
        rect_h = (maxy - miny) + height
        return cx, cy, rect_w, rect_h, 0.0
