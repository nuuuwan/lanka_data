import math

from api.utils_future.dcn.DCNUtilsCompute import DCNUtilsCompute


class DCNUtilsAlgorithm:

    @staticmethod
    def _iter_coord_lists(geometry):
        if geometry["type"] == "Polygon":
            yield from geometry["coordinates"]
        elif geometry["type"] == "MultiPolygon":
            for poly in geometry["coordinates"]:
                yield from poly

    @staticmethod
    def load_weights(features, region_id_to_weight):
        weights = {}
        for feat in features:
            fid = DCNUtilsCompute.get_feature_id(feat)
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
    def compute_force_params(
        features, areas, weights, total_value, total_area
    ):
        radius, mass, size_errors = {}, {}, []
        for feat in features:
            fid = DCNUtilsCompute.get_feature_id(feat)
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
    def _displace_coord(coord, poly_centroids, poly_radii, poly_masses, frf):
        px, py = coord[0], coord[1]
        dx = dy = 0.0
        for (cx, cy), r, m in zip(poly_centroids, poly_radii, poly_masses):
            ddx, ddy = px - cx, py - cy
            dist = math.sqrt(ddx * ddx + ddy * ddy)
            if dist == 0:
                continue
            angle = math.atan2(ddy, ddx)
            f = DCNUtilsCompute.get_fij(dist, r, m)
            dx += f * math.cos(angle)
            dy += f * math.sin(angle)
        coord[0] += dx * frf
        coord[1] += dy * frf

    @staticmethod
    def apply_forces(features, centroids, radius, mass, frf):
        ids = [DCNUtilsCompute.get_feature_id(f) for f in features]
        poly_centroids = [centroids[fid] for fid in ids]
        poly_radii = [radius[fid] for fid in ids]
        poly_masses = [mass[fid] for fid in ids]
        for feat in features:
            for coord_list in DCNUtilsAlgorithm._iter_coord_lists(
                feat["geometry"]
            ):
                for coord in coord_list:
                    DCNUtilsAlgorithm._displace_coord(
                        coord, poly_centroids, poly_radii, poly_masses, frf
                    )
