class DCNUtilsCompute:
    @staticmethod
    def get_feature_id(feature):
        return feature["properties"]["region_id"]

    @staticmethod
    def get_polygon_area_and_centroid(coords):
        n = len(coords)
        area = cx = cy = 0.0
        for i in range(n):
            x0, y0 = coords[i][0], coords[i][1]
            x1, y1 = coords[(i + 1) % n][0], coords[(i + 1) % n][1]
            cross = x0 * y1 - x1 * y0
            area += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        area /= 2.0
        if area == 0:
            xs, ys = [c[0] for c in coords], [c[1] for c in coords]
            return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)
        return abs(area), cx / (6.0 * area), cy / (6.0 * area)

    @staticmethod
    def get_feature_area_and_centroid(geometry):
        if geometry["type"] == "Polygon":
            return DCNUtilsCompute.get_polygon_area_and_centroid(
                geometry["coordinates"][0]
            )
        total_area = total_cx = total_cy = 0.0
        for poly in geometry["coordinates"]:
            a, cx, cy = DCNUtilsCompute.get_polygon_area_and_centroid(poly[0])
            total_area += a
            total_cx += a * cx
            total_cy += a * cy
        if total_area > 0:
            return total_area, total_cx / total_area, total_cy / total_area
        return 0.0, 0.0, 0.0

    @staticmethod
    def get_geometry_stats(features):
        areas, centroids = {}, {}
        for feat in features:
            fid = DCNUtilsCompute.get_feature_id(feat)
            area, cx, cy = DCNUtilsCompute.get_feature_area_and_centroid(
                feat["geometry"]
            )
            areas[fid] = area
            centroids[fid] = (cx, cy)
        return areas, centroids

    @staticmethod
    def get_fij(dist, r, m):
        if dist > r:
            return m * (r / dist)
        ratio = dist / r
        return m * (dist * dist / (r * r)) * (4.0 - 3.0 * ratio)
