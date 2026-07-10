class SquareMapLabelGeoMixin:
    @staticmethod
    def _region_squares(layout):
        squares = {}
        for region_id, x, y in layout["squares"]:
            squares.setdefault(region_id, []).append((x, y))
        return squares

    @staticmethod
    def _region_centroid(points):
        point_count = len(points)
        return (
            sum(x for x, _ in points) / point_count,
            sum(y for _, y in points) / point_count,
        )

    @classmethod
    def _snapped_position(cls, points):
        centroid = cls._region_centroid(points)
        return min(
            points,
            key=lambda point: (point[0] - centroid[0]) ** 2
            + (point[1] - centroid[1]) ** 2,
        )

    @classmethod
    def _region_positions(cls, layout, snap):
        positions = {}
        for region_id, points in cls._region_squares(layout).items():
            positions[region_id] = (
                cls._snapped_position(points)
                if snap
                else cls._region_centroid(points)
            )
        return positions
