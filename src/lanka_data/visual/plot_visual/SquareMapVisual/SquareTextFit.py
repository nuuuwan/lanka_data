class SquareTextFit:
    ADJACENCY_TOLERANCE = 0.5
    ANGLES = ((0.0, 0, 1), (90.0, 1, 0))

    @staticmethod
    def _lines(points, secondary, side):
        lines = {}
        for point in points:
            key = round(point[secondary] / side)
            lines.setdefault(key, []).append(point)
        return lines

    @classmethod
    def _longest_run(cls, items, primary, side):
        items.sort(key=lambda item: item[primary])
        longest_run = run = [items[0]]
        tolerance = side * cls.ADJACENCY_TOLERANCE
        for prev, cur in zip(items, items[1:]):
            step = cur[primary] - prev[primary]
            run = run + [cur] if abs(step - side) < tolerance else [cur]
            if len(run) > len(longest_run):
                longest_run = run
        return longest_run

    @classmethod
    def _best_run(cls, points, side):
        best = None
        for angle, primary, secondary in cls.ANGLES:
            for items in cls._lines(points, secondary, side).values():
                run = cls._longest_run(items, primary, side)
                if best is None or len(run) > len(best[0]):
                    best = (run, angle)
        return best

    @classmethod
    def best_label_fit(cls, points, size):
        side = 2 * size
        run, angle = cls._best_run(points, side)
        first, last = run[0], run[-1]
        cx = (first[0] + last[0]) / 2
        cy = (first[1] + last[1]) / 2
        return cx, cy, len(run) * side, side, angle
