import math


class HexTextFit:
    ANGLES_DEG = (0.0, 60.0, -60.0)

    @staticmethod
    def _axes(angle_deg):
        theta = math.radians(angle_deg)
        cos_a, sin_a = math.cos(theta), math.sin(theta)
        return (cos_a, sin_a), (-sin_a, cos_a)

    @staticmethod
    def _project(point, axis):
        return point[0] * axis[0] + point[1] * axis[1]

    @classmethod
    def _lines(cls, points, u, v, line_spacing):
        lines = {}
        for point in points:
            key = round(cls._project(point, v) / line_spacing)
            lines.setdefault(key, []).append((cls._project(point, u), point))
        return lines

    @staticmethod
    def _longest_in_line(items, step):
        items.sort(key=lambda item: item[0])
        best = run = [items[0]]
        for prev, cur in zip(items, items[1:]):
            run = (
                run + [cur]
                if abs(cur[0] - prev[0] - step) < step * 0.5
                else [cur]
            )
            if len(run) > len(best):
                best = run
        return best

    @classmethod
    def _best_run(cls, points, radius):
        step = math.sqrt(3) * radius
        line_spacing = 1.5 * radius
        best = None
        for angle_deg in cls.ANGLES_DEG:
            u, v = cls._axes(angle_deg)
            for items in cls._lines(points, u, v, line_spacing).values():
                run = cls._longest_in_line(items, step)
                if best is None or len(run) > best[0]:
                    best = (len(run), angle_deg, run)
        return best, step

    @staticmethod
    def _run_center(run):
        first, last = run[0][1], run[-1][1]
        return (first[0] + last[0]) / 2, (first[1] + last[1]) / 2

    @classmethod
    def best_label_fit(cls, points, radius):
        (count, angle_deg, run), step = cls._best_run(points, radius)
        cx, cy = cls._run_center(run)
        return cx, cy, count * step, 1.5 * radius, angle_deg
