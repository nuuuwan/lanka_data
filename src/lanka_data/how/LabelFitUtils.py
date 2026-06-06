from lanka_data.how.PolyUtils import PolyUtils


class LabelFitUtils:
    @staticmethod
    def _score_rect_at(rpoly, rboundary, px, py, span):
        from shapely.geometry import LineString, Point, box

        cp = Point(px, py)

        def ray(ddx, ddy):
            ln = LineString([(px, py), (px + ddx * span, py + ddy * span)])
            inter = rboundary.intersection(ln)
            if inter.is_empty:
                return span
            pts = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
            dists = [cp.distance(p) for p in pts]
            return min(dists) if dists else span

        hw = min(ray(-1, 0), ray(1, 0))
        hh = min(ray(0, -1), ray(0, 1))
        rect_geom = box(px - hw, py - hh, px + hw, py + hh)
        score = rect_geom.intersection(rpoly).area
        return hw, hh, score

    @staticmethod
    def _coarse_scan(poly, n_angles=36):
        from shapely.affinity import rotate as shapely_rotate

        pole = PolyUtils._pole_of_inaccessibility(poly)
        px0, py0 = pole.x, pole.y
        b = poly.bounds
        span = max(b[2] - b[0], b[3] - b[1]) * 2
        results = []
        for i in range(n_angles):
            angle_deg = i * 180.0 / n_angles
            rpoly = shapely_rotate(poly, -angle_deg, origin=(px0, py0))
            hw, hh, score = LabelFitUtils._score_rect_at(
                rpoly, rpoly.boundary, px0, py0, span
            )
            results.append((score, angle_deg, px0, py0, 2 * hw, 2 * hh))
        results.sort(reverse=True)
        return results, span

    @staticmethod
    def _fine_search(poly, angle_results, span, n_top=5, n_grid=6):
        import math

        from shapely.affinity import rotate as shapely_rotate
        from shapely.geometry import Point

        b = poly.bounds
        ox, oy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        candidates = PolyUtils._interior_candidates(poly, n_cells=n_grid)
        best_score, best_angle, best_cx, best_cy, best_rw, best_rh = (
            angle_results[0]
        )
        for _, angle_deg, _, _, _, _ in angle_results[:n_top]:
            theta = math.radians(-angle_deg)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            rpoly = shapely_rotate(poly, -angle_deg, origin=(ox, oy))
            rboundary = rpoly.boundary
            for px, py in candidates:
                dx, dy = px - ox, py - oy
                rpx = ox + dx * cos_t - dy * sin_t
                rpy = oy + dx * sin_t + dy * cos_t
                if not rpoly.contains(Point(rpx, rpy)):
                    continue
                hw, hh, score = LabelFitUtils._score_rect_at(
                    rpoly, rboundary, rpx, rpy, span
                )
                if score > best_score:
                    best_score = score
                    best_rw, best_rh = 2 * hw, 2 * hh
                    best_angle = angle_deg
                    dx2, dy2 = rpx - ox, rpy - oy
                    best_cx = ox + dx2 * cos_t + dy2 * sin_t
                    best_cy = oy - dx2 * sin_t + dy2 * cos_t
        return best_cx, best_cy, best_rw, best_rh, best_angle

    @staticmethod
    def _best_label_fit(geom):
        poly = PolyUtils._largest_polygon(geom)
        angle_results, span = LabelFitUtils._coarse_scan(poly)
        return LabelFitUtils._fine_search(poly, angle_results, span)
