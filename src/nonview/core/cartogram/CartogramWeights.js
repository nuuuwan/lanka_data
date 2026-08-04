import { getFeatureId } from "./CartogramGeometry.js";

export function loadWeights(features, regionIdToWeight) {
  const weights = {};
  for (const feature of features) {
    const featureId = getFeatureId(feature);
    const weight = regionIdToWeight[featureId] ?? 1;
    if (weight < 0) {
      throw new Error(`Negative PolygonValue for region ${featureId}`);
    }
    weights[featureId] = weight;
  }
  const total = Object.values(weights).reduce((sum, weight) => sum + weight, 0);
  if (total === 0) {
    throw new Error("TotalValue is zero; all weights are zero.");
  }
  return {
    weights: Object.fromEntries(
      Object.entries(weights).map(([featureId, weight]) => [
        featureId,
        weight / total,
      ]),
    ),
    totalValue: 1,
  };
}

export function computeForceParams(
  features,
  areas,
  weights,
  totalValue,
  totalArea,
) {
  const radius = {};
  const mass = {};
  const sizeErrors = [];
  for (const feature of features) {
    const featureId = getFeatureId(feature);
    const desired = totalArea * (weights[featureId] / totalValue);
    const actual = areas[featureId];
    radius[featureId] = Math.sqrt(actual / Math.PI);
    mass[featureId] =
      Math.sqrt(desired / Math.PI) - Math.sqrt(actual / Math.PI);
    const denominator = Math.min(actual, desired);
    sizeErrors.push(
      denominator > 0 ? Math.max(actual, desired) / denominator : 1,
    );
  }
  const meanSizeError =
    sizeErrors.reduce((sum, error) => sum + error, 0) / sizeErrors.length;
  return { radius, mass, frf: 1 / (1 + meanSizeError), meanSizeError };
}
