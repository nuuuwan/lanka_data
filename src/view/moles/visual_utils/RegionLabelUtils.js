import {
  MAP_MAX_LABEL_COUNT,
  MAP_UNKNOWN_COLOR,
} from "../../../nonview/constants/Map.js";
import { getBestLabelFit } from "./RegionLabelFitUtils.js";

const geometryToProjectionFits = new WeakMap();

function getFeatureLabelFit(feature, projection) {
  let projectionFits = geometryToProjectionFits.get(feature.geometry);
  if (!projectionFits) {
    projectionFits = new WeakMap();
    geometryToProjectionFits.set(feature.geometry, projectionFits);
  }
  let fit = projectionFits.get(projection);
  if (!fit) {
    fit = getBestLabelFit(feature.properties.name, feature, projection);
    if (fit) projectionFits.set(projection, fit);
  }
  return fit;
}

export function buildRegionLabels(features, projection) {
  if (features.length > MAP_MAX_LABEL_COUNT) return [];
  return features
    .map((feature) => {
      const fit = getFeatureLabelFit(feature, projection);
      return fit
        ? {
            ...fit,
            backgroundColor: feature.fill ?? MAP_UNKNOWN_COLOR,
            id: feature.id,
            name: feature.properties.name,
          }
        : null;
    })
    .filter(
      (label) =>
        label &&
        label.position.every(Number.isFinite) &&
        Number.isFinite(label.fontSize),
    );
}
