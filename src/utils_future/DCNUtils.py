"""
Implements Dougenik, Chrisman, and Niemeyer's algorithm.

```pseudocode
For each polygon
    Read and store PolygonValue (negative value illegal)
    Sum PolygonValue into TotalValue

For each iteration (user controls when done)
    For each polygon
        Calculate area and centroid (using current boundaries)
    Sum areas into TotalArea

    For each polygon
        Desired = (TotalArea * (PolygonValue / TotalValue))
        Radius = √ (Area / 𝜋)
        Mass = √ (Desired / 𝜋) - √ (Area / 𝜋)
        SizeError = Max(Area, Desired) / Min(Area, Desired)
    ForceReductionFactor = 1 / (1 + Mean (SizeError))

    For each boundary line; Read coordinate chain
        For each coordinate pair
            For each polygon centroid
                Find angle, Distance from centroid to coordinate
                    If (Distance > Radius of polygon)
                        Fij = Mas * (Radius / Distance)
                    Else
                        Fij = Mass * (Distance² / Radius²)
                              * (4 - 3 * (Distance / Radius))
            Using Fij and angles, calculate vector sum
            Multiply by ForceReductionFactor
            Move coordinate accordingly
        Write distorted line to output and plot result
```
"""

import json
import math
import time

from shapely.geometry import shape

from utils_future.Log import Log

log = Log("DCNUtils")


class DCNUtils:
    EPSILON = 0.01
    MAX_ITERATIONS = 20
    MAX_TIME = 30.0

    @staticmethod
    def _polygon_area_and_centroid(coords):
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
    def _feature_area_and_centroid(geometry):
        if geometry["type"] == "Polygon":
            return DCNUtils._polygon_area_and_centroid(
                geometry["coordinates"][0]
            )
        total_area = total_cx = total_cy = 0.0
        for poly in geometry["coordinates"]:
            a, cx, cy = DCNUtils._polygon_area_and_centroid(poly[0])
            total_area += a
            total_cx += a * cx
            total_cy += a * cy
        if total_area > 0:
            return total_area, total_cx / total_area, total_cy / total_area
        return 0.0, 0.0, 0.0

    @staticmethod
    def _get_feature_id(feature):
        return feature["properties"]["region_id"]

    @staticmethod
    def _iter_coord_lists(geometry):
        if geometry["type"] == "Polygon":
            yield from geometry["coordinates"]
        elif geometry["type"] == "MultiPolygon":
            for poly in geometry["coordinates"]:
                yield from poly

    @staticmethod
    def _load_weights(features, region_id_to_weight):
        weights = {}
        for feat in features:
            fid = DCNUtils._get_feature_id(feat)
            w = region_id_to_weight.get(fid, 1.0)
            if w < 0:
                raise ValueError(f"Negative PolygonValue for region {fid!r}")
            weights[fid] = w
        total = sum(weights.values())
        if total == 0:
            raise ValueError("TotalValue is zero; all weights are zero.")
        weights = {fid: w / total for fid, w in weights.items()}
        return weights, 1.0

    @staticmethod
    def _compute_geometry_stats(features):
        areas, centroids = {}, {}
        for feat in features:
            fid = DCNUtils._get_feature_id(feat)
            area, cx, cy = DCNUtils._feature_area_and_centroid(
                feat["geometry"]
            )
            areas[fid] = area
            centroids[fid] = (cx, cy)
        return areas, centroids

    @staticmethod
    def _compute_force_params(
        features, areas, weights, total_value, total_area
    ):
        radius, mass, size_errors = {}, {}, []
        for feat in features:
            fid = DCNUtils._get_feature_id(feat)
            desired = total_area * (weights[fid] / total_value)
            actual = areas[fid]
            r = math.sqrt(actual / math.pi)
            m = math.sqrt(desired / math.pi) - math.sqrt(actual / math.pi)
            denom = min(actual, desired)
            size_errors.append(
                max(actual, desired) / denom if denom > 0 else 1.0
            )
            radius[fid] = r
            mass[fid] = m
        mean_size_error = sum(size_errors) / len(size_errors)
        frf = 1.0 / (1.0 + mean_size_error)
        return radius, mass, frf, mean_size_error

    @staticmethod
    def _fij(dist, r, m):
        if dist > r:
            return m * (r / dist)
        ratio = dist / r
        return m * (dist * dist / (r * r)) * (4.0 - 3.0 * ratio)

    @staticmethod
    def _displace_coord(coord, poly_centroids, poly_radii, poly_masses, frf):
        px, py = coord[0], coord[1]
        dx = dy = 0.0
        for (cx, cy), r, m in zip(poly_centroids, poly_radii, poly_masses):
            ddx, ddy = px - cx, py - cy
            dist = math.sqrt(ddx * ddx + ddy * ddy)
            if dist == 0:
                continue
            angle = math.atan2(ddy, ddx)
            f = DCNUtils._fij(dist, r, m)
            dx += f * math.cos(angle)
            dy += f * math.sin(angle)
        coord[0] += dx * frf
        coord[1] += dy * frf

    @staticmethod
    def _apply_forces(features, centroids, radius, mass, frf):
        ids = [DCNUtils._get_feature_id(f) for f in features]
        poly_centroids = [centroids[fid] for fid in ids]
        poly_radii = [radius[fid] for fid in ids]
        poly_masses = [mass[fid] for fid in ids]
        for feat in features:
            for coord_list in DCNUtils._iter_coord_lists(feat["geometry"]):
                for coord in coord_list:
                    DCNUtils._displace_coord(
                        coord, poly_centroids, poly_radii, poly_masses, frf
                    )

    @staticmethod
    def _run_features(features, region_id_to_weight):
        weights, total_value = DCNUtils._load_weights(
            features, region_id_to_weight
        )
        t_start = time.time()
        for i in range(DCNUtils.MAX_ITERATIONS):
            areas, centroids = DCNUtils._compute_geometry_stats(features)
            total_area = sum(areas.values())
            if total_area == 0:
                break
            radius, mass, frf, mean_size_error = (
                DCNUtils._compute_force_params(
                    features, areas, weights, total_value, total_area
                )
            )
            error = mean_size_error - 1.0
            if error < DCNUtils.EPSILON:
                log.debug("Converged.")
                break
            td = time.time() - t_start
            if td > DCNUtils.MAX_TIME:
                log.debug("Max time reached.")
                break
            log.debug(
                f"iteration={i + 1} <= {DCNUtils.MAX_ITERATIONS},"
                + f" error={error:.4f} >= {DCNUtils.EPSILON}"
                + f" time={td:.2f}s < {DCNUtils.MAX_TIME}s"
            )

            DCNUtils._apply_forces(features, centroids, radius, mass, frf)

    @staticmethod
    def run_gdf(
        gdf,
        region_id_to_weight: dict[str, float],
    ):
        geojson = json.loads(gdf.to_json())
        features = geojson["features"]
        DCNUtils._run_features(features, region_id_to_weight)
        result = gdf.copy()
        result["geometry"] = [
            shape(f["geometry"]).buffer(0) for f in features
        ]
        return result
