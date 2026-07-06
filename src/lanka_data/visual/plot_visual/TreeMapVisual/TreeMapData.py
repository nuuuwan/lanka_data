class TreeMapData:
    @staticmethod
    def _normalize(values, dx, dy):
        total = sum(values)
        if total <= 0:
            return []
        scale = dx * dy / total
        return [v * scale for v in values]

    @staticmethod
    def _row_rects(sizes, x, y, dx, dy):
        horizontal = dx >= dy
        side = sum(sizes) / (dy if horizontal else dx)
        rects = []
        for size in sizes:
            length = size / side if side else 0
            if horizontal:
                rects.append((x, y, side, length))
                y += length
            else:
                rects.append((x, y, length, side))
                x += length
        return rects

    @classmethod
    def _worst(cls, sizes, x, y, dx, dy):
        ratios = [
            max(w / h, h / w)
            for _, _, w, h in cls._row_rects(sizes, x, y, dx, dy)
            if w > 0 and h > 0
        ]
        return max(ratios) if ratios else float("inf")

    @staticmethod
    def _leftover(sizes, x, y, dx, dy):
        if dx >= dy:
            side = sum(sizes) / dy
            return (x + side, y, dx - side, dy)
        side = sum(sizes) / dx
        return (x, y + side, dx, dy - side)

    @classmethod
    def _split(cls, sizes, x, y, dx, dy):
        i = 1
        while i < len(sizes) and cls._worst(
            sizes[:i], x, y, dx, dy
        ) >= cls._worst(sizes[: i + 1], x, y, dx, dy):
            i += 1
        return i

    @classmethod
    def _squarify(cls, sizes, x, y, dx, dy):
        if not sizes:
            return []
        if len(sizes) == 1:
            return cls._row_rects(sizes, x, y, dx, dy)
        i = cls._split(sizes, x, y, dx, dy)
        lx, ly, ldx, ldy = cls._leftover(sizes[:i], x, y, dx, dy)
        return cls._row_rects(sizes[:i], x, y, dx, dy) + cls._squarify(
            sizes[i:], lx, ly, ldx, ldy
        )

    @classmethod
    def layout(cls, values, x, y, dx, dy):
        sizes = cls._normalize([v for v in values if v > 0], dx, dy)
        return cls._squarify(sizes, x, y, dx, dy)
