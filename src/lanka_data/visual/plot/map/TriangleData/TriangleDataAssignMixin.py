from lanka_data.visual.plot.map.HexData.HexDataAssignMixin import (
    HexDataAssignMixin,
)
from utils_future import HungarianUtils


class TriangleDataAssignMixin(HexDataAssignMixin):
    @staticmethod
    def _cost_matrix(slots, centers):
        cost = []
        for _, (cx, cy) in slots:
            row = [(cx - x) ** 2 + (cy - y) ** 2 for (x, y, _) in centers]
            cost.append(row)
        return cost

    @classmethod
    def assign(cls, region_to_centroid, counts, centers):
        slots = cls._build_slots(region_to_centroid, counts)
        cost = cls._cost_matrix(slots, centers)
        assignment = HungarianUtils.solve(cost)
        triangles = []
        for i, j in enumerate(assignment):
            if j < 0:
                continue
            region_id = slots[i][0]
            x, y, up = centers[j]
            triangles.append([region_id, x, y, up])
        return triangles
