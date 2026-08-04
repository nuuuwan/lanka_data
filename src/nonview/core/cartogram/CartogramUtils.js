import { applyForces } from "./CartogramForces.js";
import { getGeometryStats } from "./CartogramGeometry.js";
import { computeForceParams, loadWeights } from "./CartogramWeights.js";

export default class CartogramUtils {
  static EPSILON = 0.01;
  static MAX_ITERATIONS = 20;
  static MAX_TIME_MS = 10000;

  static compute(features, regionIdToWeight) {
    const { weights, totalValue } = loadWeights(features, regionIdToWeight);
    const startTime = performance.now();
    for (
      let iteration = 0;
      iteration < CartogramUtils.MAX_ITERATIONS;
      iteration += 1
    ) {
      const { areas, centroids } = getGeometryStats(features);
      const totalArea = Object.values(areas).reduce(
        (sum, area) => sum + area,
        0,
      );
      if (totalArea === 0) {
        break;
      }
      const { radius, mass, frf, meanSizeError } = computeForceParams(
        features,
        areas,
        weights,
        totalValue,
        totalArea,
      );
      if (meanSizeError - 1 < CartogramUtils.EPSILON) {
        break;
      }
      if (performance.now() - startTime > CartogramUtils.MAX_TIME_MS) {
        break;
      }
      applyForces(features, centroids, radius, mass, frf);
    }
  }
}
