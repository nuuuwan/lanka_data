import { getFeatureId } from "./CartogramGeometry.js";

export function loadWeights(features, regionIdToWeight) {
  const weights = {};
  for (const feature of features) {
    const fid = getFeatureId(feature);
    const w = regionIdToWeight[fid] ?? 1;
    if (w < 0) {
      throw new Error(`Negative PolygonValue for region ${fid}`);
    }
    weights[fid] = w;
  }

  const total = Object.values(weights).reduce((sum, w) => sum + w, 0);
  if (total === 0) {
    throw new Error("TotalValue is zero; all weights are zero.");
  }

  const normalized = {};
  for (const [fid, w] of Object.entries(weights)) {
    normalized[fid] = w / total;
  }
  return { weights: normalized, totalValue: 1 };
}
