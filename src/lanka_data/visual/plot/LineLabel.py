import math

from shapely.geometry import MultiLineString, Point

from lanka_data.visual.plot.LabelTruncator import LabelTruncator
from utils_future import timer


class LineLabel:
    FONT_SIZE = 6
    TEXT_COLOR = "#333"
    SEA_OFFSET_RATIO = 0.08
    INLAND_OFFSET_RATIO = 0.1

    @staticmethod
    def _longest_line(geom):
        if isinstance(geom, MultiLineString):
            if geom.is_empty:
                return geom
            return max(geom.geoms, key=lambda line: line.length)
        return geom

    @staticmethod
    def _normalize_angle(angle):
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        return angle

    @classmethod
    def _mouth_and_inland(cls, line, center):
        start = line.interpolate(0.0, normalized=True)
        end = line.interpolate(1.0, normalized=True)
        inland_ratio = cls.INLAND_OFFSET_RATIO
        if start.distance(center) >= end.distance(center):
            return start, line.interpolate(inland_ratio, normalized=True)
        return end, line.interpolate(1.0 - inland_ratio, normalized=True)

    @classmethod
    def _label_placement(cls, line, center):
        mouth, inland = cls._mouth_and_inland(line, center)
        dx, dy = mouth.x - inland.x, mouth.y - inland.y
        norm = math.hypot(dx, dy)
        if norm == 0:
            return mouth.x, mouth.y, 0.0
        offset = cls.SEA_OFFSET_RATIO * line.length
        px = mouth.x + dx / norm * offset
        py = mouth.y + dy / norm * offset
        angle = cls._normalize_angle(math.degrees(math.atan2(dy, dx)))
        return px, py, angle

    @classmethod
    def _draw_one(cls, row, ax, region_count, center):
        name = row.get("region_name") or str(row.get("region_id"))
        label = LabelTruncator.get_label(name, region_count)
        if label is None:
            return
        line = cls._longest_line(row.geometry)
        if line.is_empty or line.length == 0:
            return
        cx, cy, angle = cls._label_placement(line, center)
        ax.annotate(
            label,
            xy=(cx, cy),
            ha="center",
            va="center",
            fontsize=cls.FONT_SIZE,
            color=cls.TEXT_COLOR,
            rotation=angle,
        )

    @staticmethod
    def _named_rows(gdf_region):
        return [
            row
            for _, row in gdf_region.iterrows()
            if row.get("is_named", True)
        ]

    @staticmethod
    def _center(gdf_region):
        min_x, min_y, max_x, max_y = gdf_region.total_bounds
        return Point((min_x + max_x) / 2, (min_y + max_y) / 2)

    @classmethod
    @timer
    def draw(cls, gdf_region, ax, region_count):
        rows = cls._named_rows(gdf_region)
        center = cls._center(gdf_region)
        for row in rows:
            cls._draw_one(row, ax, len(rows), center)
