import StringUtils from "../../../nonview/base/String.js";
import {
  MAP_LABEL_MIN_FONT_SIZE,
  MAP_MAX_LABEL_COUNT,
  MAP_UNKNOWN_COLOR,
} from "../../_cons/MapCons.js";
import { getBestLabelFit } from "./RegionLabelFitUtils.js";

const geometryToProjectionFits = new WeakMap();

function getFeatureLabelFit(feature, projection, name) {
  let projectionFits = geometryToProjectionFits.get(feature.geometry);
  if (!projectionFits) {
    projectionFits = new WeakMap();
    geometryToProjectionFits.set(feature.geometry, projectionFits);
  }
  let fitsByName = projectionFits.get(projection);
  if (!fitsByName) {
    fitsByName = new Map();
    projectionFits.set(projection, fitsByName);
  }
  let fit = fitsByName.get(name);
  if (!fit) {
    fit = getBestLabelFit(name, feature, projection);
    if (fit) fitsByName.set(name, fit);
  }
  return fit;
}

function getDisplayLabel(feature, projection) {
  const fullName = feature.properties.name;
  const fullFit = getFeatureLabelFit(feature, projection, fullName);
  if (!fullFit) {
    return null;
  }
  let best = { fit: fullFit, name: fullName };
  if (fullFit.fontSize >= MAP_LABEL_MIN_FONT_SIZE) {
    return best;
  }
  for (const maxLen of [3, 2, 1]) {
    const shortName = StringUtils.shorten(fullName, maxLen);
    if (shortName === best.name) {
      continue;
    }
    const shortFit = getFeatureLabelFit(feature, projection, shortName);
    if (shortFit && shortFit.fontSize > best.fit.fontSize) {
      best = { fit: shortFit, name: shortName };
    }
  }
  return best;
}

export function buildRegionLabels(features, projection) {
  if (features.length > MAP_MAX_LABEL_COUNT) return [];
  return features
    .map((feature) => {
      const displayLabel = getDisplayLabel(feature, projection);
      return displayLabel
        ? {
            ...displayLabel.fit,
            backgroundColor: feature.fill ?? MAP_UNKNOWN_COLOR,
            id: feature.id,
            name: displayLabel.name,
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
