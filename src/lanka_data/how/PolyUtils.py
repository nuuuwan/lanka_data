class PolyUtils:
    @staticmethod
    def _collect_polygons(g):
        from shapely.geometry import Polygon

        if isinstance(g, Polygon):
            return [g]
        if hasattr(g, "geoms"):
            polys = []
            for sub in g.geoms:
                polys.extend(PolyUtils._collect_polygons(sub))
            return polys
        return []

    @staticmethod
    def _largest_polygon(geom):
        polys = PolyUtils._collect_polygons(geom)
        if not polys:
            return geom
        return max(polys, key=lambda g: g.area)

    @staticmethod
    def _pole_of_inaccessibility(poly, n_cells=32):
        from shapely.geometry import Point

        minx, miny, maxx, maxy = poly.bounds
        boundary = poly.boundary
        best_dist = -1.0
        best_pt = poly.representative_point()
        xs = [
            minx + (maxx - minx) * (i + 0.5) / n_cells for i in range(n_cells)
        ]
        ys = [
            miny + (maxy - miny) * (j + 0.5) / n_cells for j in range(n_cells)
        ]
        for x in xs:
            for y in ys:
                pt = Point(x, y)
                if poly.contains(pt):
                    d = boundary.distance(pt)
                    if d > best_dist:
                        best_dist = d
                        best_pt = pt
        return best_pt

    @staticmethod
    def _interior_candidates(poly, n_cells=6):
        from shapely.geometry import Point

        minx, miny, maxx, maxy = poly.bounds
        pts = []
        for gi in range(n_cells):
            for gj in range(n_cells):
                x = minx + (maxx - minx) * (gi + 0.5) / n_cells
                y = miny + (maxy - miny) * (gj + 0.5) / n_cells
                if poly.contains(Point(x, y)):
                    pts.append((x, y))
        if not pts:
            rp = poly.representative_point()
            pts = [(rp.x, rp.y)]
        return pts
