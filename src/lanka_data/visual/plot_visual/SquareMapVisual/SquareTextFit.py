class SquareTextFit:
    ADJACENCY_TOLERANCE = 0.5

    @staticmethod
    def _rows(points, side):
        rows = {}
        for point in points:
            key = round(point[1] / side)
            rows.setdefault(key, []).append(point)
        return rows

    @classmethod
    def _longest_run(cls, items, side):
        items.sort(key=lambda item: item[0])
        longest_run = run = [items[0]]
        tolerance = side * cls.ADJACENCY_TOLERANCE
        for prev, cur in zip(items, items[1:]):
            run = (
                run + [cur]
                if abs(cur[0] - prev[0] - side) < tolerance
                else [cur]
            )
            if len(run) > len(longest_run):
                longest_run = run
        return longest_run

    @classmethod
    def best_label_fit(cls, points, size):
        side = 2 * size
        best = None
        for items in cls._rows(points, side).values():
            run = cls._longest_run(list(items), side)
            if best is None or len(run) > len(best):
                best = run
        first, last = best[0], best[-1]
        cx = (first[0] + last[0]) / 2
        cy = first[1]
        return cx, cy, len(best) * side, side, 0.0
