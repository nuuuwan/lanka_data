import { applyForces, computeForceParams } from "./CartogramForces.js";
import { getGeometryStats } from "./CartogramGeometry.js";
import { loadWeights } from "./CartogramWeights.js";

export default class CartogramUtils {
  static EPSILON = 0.01;
  static MAX_ITERATIONS = 20;
  static MAX_TIME_MS = 10000;

  static compute(features, regionIdToWeight) {
    const { weights, totalValue } = loadWeights(features, regionIdToWeight);
    const startTime = performance.now();

    for (let i = 0; i < CartogramUtils.MAX_ITERATIONS; i++) {
      const { areas, centroids } = getGeometryStats(features);
      const totalArea = Object.values(areas).reduce((sum, a) => sum + a, 0);
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

      const error = meanSizeError - 1;
      if (error < CartogramUtils.EPSILON) {
        break;
      }

      if (performance.now() - startTime > CartogramUtils.MAX_TIME_MS) {
        break;
      }

      applyForces(features, centroids, radius, mass, frf);
    }
  }
}
