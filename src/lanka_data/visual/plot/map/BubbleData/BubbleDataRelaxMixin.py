import math


class BubbleDataRelaxMixin:
    ITERATIONS = 500
    PADDING_RATIO = 0.01
    GOLDEN_ANGLE = 2.399963

    @staticmethod
    def _padding(bounds):
        minx, miny, maxx, maxy = bounds
        span = max(maxx - minx, maxy - miny)
        return span * BubbleDataRelaxMixin.PADDING_RATIO

    @staticmethod
    def _offset(pos_i, pos_j, index):
        dx = pos_j[0] - pos_i[0]
        dy = pos_j[1] - pos_i[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            angle = index * BubbleDataRelaxMixin.GOLDEN_ANGLE
            return math.cos(angle), math.sin(angle), 1e-9
        return dx / dist, dy / dist, dist

    @classmethod
    def _resolve_pair(cls, pos, radii, pad, ids, i, j):
        id_i, id_j = ids[i], ids[j]
        nx, ny, dist = cls._offset(pos[id_i], pos[id_j], i + j)
        min_dist = radii[id_i] + radii[id_j] + pad
        if dist >= min_dist:
            return False
        shift = (min_dist - dist) / 2
        pos[id_i] = (pos[id_i][0] - nx * shift, pos[id_i][1] - ny * shift)
        pos[id_j] = (pos[id_j][0] + nx * shift, pos[id_j][1] + ny * shift)
        return True

    @classmethod
    def _iterate(cls, pos, radii, pad, ids):
        moved = False
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if cls._resolve_pair(pos, radii, pad, ids, i, j):
                    moved = True
        return moved

    @classmethod
    def relax(cls, centroids, radii, bounds):
        pos = dict(centroids)
        ids = list(centroids.keys())
        pad = cls._padding(bounds)
        for _ in range(cls.ITERATIONS):
            if not cls._iterate(pos, radii, pad, ids):
                break
        return [
            (
                region_id,
                pos[region_id][0],
                pos[region_id][1],
                radii[region_id],
            )
            for region_id in ids
        ]
