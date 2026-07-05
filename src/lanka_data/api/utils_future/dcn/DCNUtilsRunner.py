import json
import time

from shapely.geometry import shape

from lanka_data.api.utils_future.dcn.DCNUtilsAlgorithm import (
    DCNUtilsAlgorithm,
)
from lanka_data.api.utils_future.dcn.DCNUtilsCompute import DCNUtilsCompute
from lanka_data.api.utils_future.Log import Log

log = Log("DCNUtilsRunner")


class DCNUtilsRunner:
    EPSILON = 0.01
    MAX_ITERATIONS = 20
    MAX_TIME = 10.0

    @staticmethod
    def _run_features(features, region_id_to_weight):
        weights, total_value = DCNUtilsAlgorithm.load_weights(
            features, region_id_to_weight
        )
        t_start = time.time()
        for i in range(DCNUtilsRunner.MAX_ITERATIONS):
            areas, centroids = DCNUtilsCompute.get_geometry_stats(features)
            total_area = sum(areas.values())
            if total_area == 0:
                break
            radius, mass, frf, mean_size_error = (
                DCNUtilsAlgorithm.compute_force_params(
                    features, areas, weights, total_value, total_area
                )
            )
            error = mean_size_error - 1.0
            if error < DCNUtilsRunner.EPSILON:
                log.debug("Converged.")
                break
            td = time.time() - t_start
            if td > DCNUtilsRunner.MAX_TIME:
                log.debug("Max time reached.")
                break
            log.debug(
                f"iteration={i + 1} <= {DCNUtilsRunner.MAX_ITERATIONS},"
                + f" error={error:.4f} >= {DCNUtilsRunner.EPSILON}"
                + f" time={td:.2f}s < {DCNUtilsRunner.MAX_TIME}s"
            )

            DCNUtilsAlgorithm.apply_forces(
                features, centroids, radius, mass, frf
            )

    @staticmethod
    def run_gdf(
        gdf,
        region_id_to_weight: dict[str, float],
    ):
        geojson = json.loads(gdf.to_json())
        features = geojson["features"]
        DCNUtilsRunner._run_features(features, region_id_to_weight)
        result = gdf.copy()
        result["geometry"] = [
            shape(f["geometry"]).buffer(0) for f in features
        ]
        return result
