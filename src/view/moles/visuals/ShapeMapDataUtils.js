import {
  buildFeatureToDataMap,
  getFeatureRegionId,
  matchFeatureToValue,
} from "../visual_utils/GeoVisualUtils.js";

function getDisplayItem(items) {
  return items.reduce((best, item) => (item.value > best.value ? item : best));
}

export function buildShapeMapFacetInfo(features, facetKey, dataMap) {
  const regions = features
    .map((feature) => {
      const match = matchFeatureToValue(feature, dataMap);
      return match
        ? {
            display: getDisplayItem(match.items),
            feature,
            id: String(getFeatureRegionId(feature)),
            weight: match.items.reduce((sum, item) => sum + item.value, 0),
          }
        : null;
    })
    .filter(Boolean);
  return { facetKey, regions };
}

export function getLegendItems(facetInfos) {
  const items = new Map();
  facetInfos.forEach(({ regions }) =>
    regions.forEach(({ display }) =>
      items.set(display.label, {
        id: display.label,
        label: display.label,
        color: display.color,
      }),
    ),
  );
  return [...items.values()];
}

export function getFacetColor(facetInfo) {
  const colors = new Set(
    facetInfo.regions.map(({ display }) => display.color).filter(Boolean),
  );
  return colors.size === 1 ? [...colors][0] : null;
}

export function getMatchedFeatures(
  geoJson,
  datumList,
  regionIndex,
  stackIndex,
) {
  const dataMap = buildFeatureToDataMap(datumList, regionIndex, stackIndex);
  return geoJson.features.filter((feature) =>
    matchFeatureToValue(feature, dataMap),
  );
}
