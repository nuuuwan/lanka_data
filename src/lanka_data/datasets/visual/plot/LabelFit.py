import math

import numpy as np

from lanka_data.api.utils_future import timer
from lanka_data.api.utils_future.PolygonUtils import PolygonUtils


class LabelFit:
    INTERIOR_CANDIDATES = getattr(PolygonUtils, "_interior_candidates")
    LARGEST_POLYGON = getattr(PolygonUtils, "_largest_polygon")

    @staticmethod
    def _edges(poly):
        coords = np.array(poly.exterior.coords)
        ax, ay = coords[:-1, 0], coords[:-1, 1]
        bx, by = coords[1:, 0], coords[1:, 1]
        return ax, ay, bx - ax, by - ay

    @staticmethod
    def _ray_dist(cx, cy, dx, dy, edges, span):
        ax, ay, ex, ey = edges
        denom = dx * ey - dy * ex
        valid = np.abs(denom) > 1e-12
        safe = np.where(valid, denom, 1.0)
        t = np.where(
            valid,
            ((ax - cx) * ey - (ay - cy) * ex) / safe,
            np.inf,
        )
        s = np.where(
            valid,
            ((ax - cx) * dy - (ay - cy) * dx) / safe,
            np.inf,
        )
        hit = valid & (t > 1e-9) & (s >= -1e-9) & (s <= 1.0 + 1e-9)
        t_hit = np.where(hit, t, np.inf)
        d = float(t_hit.min())
        return d if np.isfinite(d) else span

    @staticmethod
    def _hw_hh(cx, cy, cos_a, sin_a, edges, span):
        rd = LabelFit._ray_dist
        hw = min(
            rd(cx, cy, cos_a, sin_a, edges, span),
            rd(cx, cy, -cos_a, -sin_a, edges, span),
        )
        hh = min(
            rd(cx, cy, -sin_a, cos_a, edges, span),
            rd(cx, cy, sin_a, -cos_a, edges, span),
        )
        return hw, hh

    @classmethod
    def _coarse_scan(cls, poly, n_angles=18):
        edges = cls._edges(poly)
        b = poly.bounds
        span = max(b[2] - b[0], b[3] - b[1]) * 2
        rp = poly.representative_point()
        px0, py0 = rp.x, rp.y
        results = []
        for i in range(n_angles):
            angle_deg = i * 180.0 / n_angles
            theta = math.radians(angle_deg)
            cos_a, sin_a = math.cos(theta), math.sin(theta)
            hw, hh = cls._hw_hh(px0, py0, cos_a, sin_a, edges, span)
            results.append((4 * hw * hh, angle_deg, px0, py0, 2 * hw, 2 * hh))
        results.sort(reverse=True)
        return results, span

    @classmethod
    def _fine_search(cls, poly, angle_results, span, n_top=3, n_grid=6):
        edges = cls._edges(poly)
        candidates = cls.INTERIOR_CANDIDATES(poly, n_cells=n_grid)
        best_score, best_angle, best_cx, best_cy, best_rw, best_rh = (
            angle_results[0]
        )
        for _, angle_deg, _, _, _, _ in angle_results[:n_top]:
            theta = math.radians(angle_deg)
            cos_a, sin_a = math.cos(theta), math.sin(theta)
            for px, py in candidates:
                hw, hh = cls._hw_hh(px, py, cos_a, sin_a, edges, span)
                score = 4 * hw * hh
                if score > best_score:
                    best_score = score
                    best_rw, best_rh = 2 * hw, 2 * hh
                    best_angle = angle_deg
                    best_cx, best_cy = px, py
        return best_cx, best_cy, best_rw, best_rh, best_angle

    @classmethod
    @timer
    def best_label_fit(cls, geom):
        poly = cls.LARGEST_POLYGON(geom)
        angle_results, span = cls._coarse_scan(poly)
        return cls._fine_search(poly, angle_results, span)
