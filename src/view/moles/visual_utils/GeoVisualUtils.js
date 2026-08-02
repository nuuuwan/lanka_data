import { geoMercator } from "d3-geo";

import StringUtils from "../../../nonview/base/String.js";
import Region from "../../../nonview/core/thing/concept/category_concept/region/region/Region.js";
import {
  MAP_HEIGHT,
  MAP_LABEL_CHARACTER_WIDTH_RATIO,
  MAP_MAX_LABEL_COUNT,
  MAP_PADDING,
  MAP_UNKNOWN_COLOR,
  MAP_WIDTH,
} from "../../_cons/MapCons.js";
import DimensionUtils from "./DimensionUtils.js";
import FormatUtils from "./FormatUtils.js";

export function getGeoDimInfo(datumList) {
  const regionDimIndex = datumList[0].query.dimThingList.findIndex(
    (thing) => thing instanceof Region,
  );
  const { varyingDimIndexes } = DimensionUtils.getDimIndexInfo(datumList);
  return {
    regionDimIndex,
    regionClass: datumList[0].query.dimThingList[regionDimIndex].constructor,
    stackDimIndex: varyingDimIndexes
      .filter((dimIndex) => dimIndex !== regionDimIndex)
      .at(-1),
  };
}

export function buildFeatureToDataMap(
  datumList,
  regionDimIndex,
  stackDimIndex,
) {
  const dataMap = new Map();
  for (const datum of datumList) {
    const regionValue = datum.query.dimThingList[regionDimIndex].value;
    if (!dataMap.has(regionValue)) {
      dataMap.set(regionValue, []);
    }
    const stackThing = datum.query.dimThingList[stackDimIndex];
    dataMap.get(regionValue).push({
      label: stackThing ? FormatUtils.toThingLabel(stackThing) : "value",
      value: parseFloat(datum.answerThing.value) || 0,
      color: stackThing
        ? stackThing.getColor()
        : datum.query.dimThingList[regionDimIndex].getColor(),
    });
  }
  return dataMap;
}

export function groupDatumListByFacet(datumList, facetDimIndexes) {
  const groups = new Map();
  for (const datum of datumList) {
    const facetKey = DimensionUtils.getFacetKey(datum, facetDimIndexes);
    if (!groups.has(facetKey)) {
      groups.set(facetKey, []);
    }
    groups.get(facetKey).push(datum);
  }
  return Array.from(groups.entries()).map(([facetKey, facetDatumList]) => ({
    facetKey,
    facetDatumList,
  }));
}

export function matchFeatureToValue(feature, dataMap) {
  const featureName = StringUtils.toSnakeCase(feature.properties.name);
  const compactFeatureName = featureName.replace(/_/g, "");
  for (const [regionValue, items] of dataMap) {
    const normalizedRegionValue = StringUtils.toSnakeCase(regionValue);
    if (
      normalizedRegionValue === featureName ||
      normalizedRegionValue.replace(/_/g, "") === compactFeatureName
    ) {
      return { regionValue, items };
    }
  }
  return null;
}

export function getFeatureRegionId(feature) {
  return feature.properties.id ?? feature.properties.name;
}

export function buildGeoVisualData(features, dataMap, legendItemMap) {
  const visualFeatures = [];
  const data = [];
  for (const geoFeature of features) {
    const match = matchFeatureToValue(geoFeature, dataMap);
    const display = match
      ? match.items.reduce((best, item) =>
          item.value > best.value ? item : best,
        )
      : null;
    const id = String(getFeatureRegionId(geoFeature));
    visualFeatures.push({
      ...geoFeature,
      id,
      fill: display?.color,
    });
    if (display) {
      data.push({ id, value: display.value, categoryLabel: display.label });
      legendItemMap.set(display.label, {
        id: display.label,
        label: display.label,
        color: display.color,
      });
    }
  }
  return { features: visualFeatures, data };
}

export function getGeoCoordinates(features) {
  const coordinates = [];
  function collect(value) {
    if (!Array.isArray(value)) {
      return;
    }
    if (typeof value[0] === "number") {
      coordinates.push(value);
      return;
    }
    value.forEach(collect);
  }
  features.forEach(({ geometry }) => collect(geometry.coordinates));
  return coordinates;
}

export function getProjectionInfo(features) {
  const projection = geoMercator().fitExtent(
    [
      [MAP_PADDING, MAP_PADDING],
      [MAP_WIDTH - MAP_PADDING, MAP_HEIGHT - MAP_PADDING],
    ],
    { type: "MultiPoint", coordinates: getGeoCoordinates(features) },
  );
  const [translateX, translateY] = projection.translate();
  return {
    projection,
    projectionScale: projection.scale(),
    projectionTranslation: [translateX / MAP_WIDTH, translateY / MAP_HEIGHT],
  };
}

export function getFittedLabelFontSize(label, width, height) {
  const estimatedWidthAtUnitSize =
    Math.max(label.length, 1) * MAP_LABEL_CHARACTER_WIDTH_RATIO;
  return Math.min(height, width / estimatedWidthAtUnitSize);
}

export function buildRegionLabels(features, projection) {
  if (features.length > MAP_MAX_LABEL_COUNT) {
    return [];
  }
  return features
    .map((feature) => {
      const bounds = getGeoCoordinates([feature])
        .map(projection)
        .reduce(
          ([minX, minY, maxX, maxY], [x, y]) => [
            Math.min(minX, x),
            Math.min(minY, y),
            Math.max(maxX, x),
            Math.max(maxY, y),
          ],
          [Infinity, Infinity, -Infinity, -Infinity],
        );
      return {
        backgroundColor: feature.fill ?? MAP_UNKNOWN_COLOR,
        fontSize: getFittedLabelFontSize(
          feature.properties.name,
          bounds[2] - bounds[0],
          bounds[3] - bounds[1],
        ),
        id: feature.id,
        name: feature.properties.name,
        position: [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2],
      };
    })
    .filter(({ position }) => position.every(Number.isFinite));
}
