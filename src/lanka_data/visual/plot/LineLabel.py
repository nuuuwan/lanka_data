import math

from shapely.geometry import MultiLineString

from lanka_data.visual.plot.LabelTruncator import LabelTruncator
from utils_future import timer


class LineLabel:
    FONT_SIZE = 6
    TEXT_COLOR = "#333"

    @staticmethod
    def _longest_line(geom):
        if isinstance(geom, MultiLineString):
            if geom.is_empty:
                return geom
            return max(geom.geoms, key=lambda line: line.length)
        return geom

    @staticmethod
    def _point_and_angle(line):
        mid = line.interpolate(0.5, normalized=True)
        p0 = line.interpolate(0.45, normalized=True)
        p1 = line.interpolate(0.55, normalized=True)
        angle = math.degrees(math.atan2(p1.y - p0.y, p1.x - p0.x))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        return mid.x, mid.y, angle

    @classmethod
    def _draw_one(cls, row, ax, region_count):
        name = row.get("region_name") or str(row.get("region_id"))
        label = LabelTruncator.get_label(name, region_count)
        if label is None:
            return
        line = cls._longest_line(row.geometry)
        if line.is_empty or line.length == 0:
            return
        cx, cy, angle = cls._point_and_angle(line)
        ax.annotate(
            label,
            xy=(cx, cy),
            ha="center",
            va="center",
            fontsize=cls.FONT_SIZE,
            color=cls.TEXT_COLOR,
            rotation=angle,
        )

    @classmethod
    @timer
    def draw(cls, gdf_region, ax, region_count):
        rows = [
            row
            for _, row in gdf_region.iterrows()
            if row.get("is_named", True)
        ]
        for row in rows:
            cls._draw_one(row, ax, len(rows))
