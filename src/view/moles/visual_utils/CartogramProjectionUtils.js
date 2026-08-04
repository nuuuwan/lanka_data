import { MAP_HEIGHT, MAP_WIDTH } from "../../../nonview/constants/Map.js";
import { getFeatureRegionId, matchFeatureToValue } from "./GeoVisualUtils.js";
export function buildRegionIdToWeight(features, dataMap) {
  const regionIdToWeight = {};
  for (const geoFeature of features) {
    const match = matchFeatureToValue(geoFeature, dataMap);
    if (match) {
      regionIdToWeight[getFeatureRegionId(geoFeature)] = match.items.reduce(
        (total, item) => total + item.value,
        0,
      );
    }
  }
  return regionIdToWeight;
}

export function getGlobalAreaProjectionScales(cartograms) {
  const maxTotal = Math.max(...cartograms.map(({ total }) => total), 0);
  if (maxTotal === 0) {
    return cartograms.map(() => 0);
  }

  const areaScaleFactors = cartograms.map(({ total }) =>
    Math.sqrt(total / maxTotal),
  );
  const globalProjectionScale = Math.min(
    ...cartograms
      .map(({ projectionScale }, index) => {
        const areaScaleFactor = areaScaleFactors[index];
        return areaScaleFactor > 0
          ? projectionScale / areaScaleFactor
          : Infinity;
      })
      .filter(Number.isFinite),
  );

  return areaScaleFactors.map(
    (areaScaleFactor) => globalProjectionScale * areaScaleFactor,
  );
}

export function getScaledProjectionTranslation(
  projectionTranslation,
  scaleRatio,
) {
  return projectionTranslation.map(
    (translation) => 0.5 + (translation - 0.5) * scaleRatio,
  );
}

export function fitCartogramProjection(cartogram, projectionScale) {
  const { projection, ...cartogramData } = cartogram;
  const projectionTranslation = getScaledProjectionTranslation(
    cartogram.projectionTranslation,
    projectionScale / cartogram.projectionScale,
  );
  projection
    .scale(projectionScale)
    .translate([
      projectionTranslation[0] * MAP_WIDTH,
      projectionTranslation[1] * MAP_HEIGHT,
    ]);
  return {
    ...cartogramData,
    projection,
    projectionScale,
    projectionTranslation,
  };
}
