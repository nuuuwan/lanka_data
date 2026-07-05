from utils_future import HungarianUtils


class HexDataAssignMixin:
    @staticmethod
    def _build_slots(region_to_centroid, counts):
        slots = []
        for region_id, count in counts.items():
            centroid = region_to_centroid[region_id]
            for _ in range(count):
                slots.append((region_id, centroid))
        return slots

    @staticmethod
    def _cost_matrix(slots, centers):
        cost = []
        for _, (cx, cy) in slots:
            row = [(cx - x) ** 2 + (cy - y) ** 2 for (x, y) in centers]
            cost.append(row)
        return cost

    @classmethod
    def assign(cls, region_to_centroid, counts, centers):
        slots = cls._build_slots(region_to_centroid, counts)
        cost = cls._cost_matrix(slots, centers)
        assignment = HungarianUtils.solve(cost)
        hexes = []
        for i, j in enumerate(assignment):
            if j < 0:
                continue
            region_id = slots[i][0]
            x, y = centers[j]
            hexes.append([region_id, x, y])
        return hexes
