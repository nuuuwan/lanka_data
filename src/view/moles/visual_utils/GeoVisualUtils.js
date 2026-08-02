import { geoMercator } from "d3-geo";

import StringUtils from "../../../nonview/base/String.js";
import Region from "../../../nonview/core/thing/concept/category_concept/region/region/Region.js";
import {
  MAP_HEIGHT,
  MAP_LABEL_CHARACTER_WIDTH_RATIO,
  MAP_LABEL_MARGIN_RATIO,
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

const LABEL_ANGLES = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165];
const LABEL_GRID_SIZE = 12;
const LABEL_FIT_ITERATIONS = 16;

function getProjectedPolygons(feature, projection) {
  const projectRing = (ring) => ring.map(projection);
  if (feature.geometry.type === "Polygon") {
    return [feature.geometry.coordinates.map(projectRing)];
  }
  if (feature.geometry.type === "MultiPolygon") {
    return feature.geometry.coordinates.map((polygon) =>
      polygon.map(projectRing),
    );
  }
  return [];
}

function getRingBounds(ring) {
  return ring.reduce(
    ([minX, minY, maxX, maxY], [x, y]) => [
      Math.min(minX, x),
      Math.min(minY, y),
      Math.max(maxX, x),
      Math.max(maxY, y),
    ],
    [Infinity, Infinity, -Infinity, -Infinity],
  );
}

function isPointOnSegment([px, py], [ax, ay], [bx, by]) {
  const crossProduct = (px - ax) * (by - ay) - (py - ay) * (bx - ax);
  if (Math.abs(crossProduct) > 1e-7) {
    return false;
  }
  return (
    px >= Math.min(ax, bx) - 1e-7 &&
    px <= Math.max(ax, bx) + 1e-7 &&
    py >= Math.min(ay, by) - 1e-7 &&
    py <= Math.max(ay, by) + 1e-7
  );
}

function isPointInRing(point, ring) {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; index += 1) {
    const start = ring[previous];
    const end = ring[index];
    if (isPointOnSegment(point, start, end)) {
      return true;
    }
    if (
      start[1] > point[1] !== end[1] > point[1] &&
      point[0] <
        ((end[0] - start[0]) * (point[1] - start[1])) /
          (end[1] - start[1]) +
          start[0]
    ) {
      inside = !inside;
    }
    previous = index;
  }
  return inside;
}

function isPointInPolygon(point, polygon) {
  return (
    isPointInRing(point, polygon[0]) &&
    polygon.slice(1).every((hole) => !isPointInRing(point, hole))
  );
}

function getRectangle(center, width, height, angle) {
  const radians = (angle * Math.PI) / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  return [
    [-width / 2, -height / 2],
    [width / 2, -height / 2],
    [width / 2, height / 2],
    [-width / 2, height / 2],
  ].map(([x, y]) => [
    center[0] + x * cosine - y * sine,
    center[1] + x * sine + y * cosine,
  ]);
}

function getCrossProduct([ax, ay], [bx, by], [cx, cy]) {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

function segmentsProperlyIntersect(startA, endA, startB, endB) {
  const crossA = getCrossProduct(startA, endA, startB);
  const crossB = getCrossProduct(startA, endA, endB);
  const crossC = getCrossProduct(startB, endB, startA);
  const crossD = getCrossProduct(startB, endB, endA);
  return crossA * crossB < -1e-7 && crossC * crossD < -1e-7;
}

function doesRectangleFit(rectangle, polygon) {
  if (!rectangle.every((point) => isPointInPolygon(point, polygon))) {
    return false;
  }
  const rectangleEdges = rectangle.map((start, index) => [
    start,
    rectangle[(index + 1) % rectangle.length],
  ]);
  return polygon.every((ring) =>
    ring.every((start, index) => {
      const end = ring[(index + 1) % ring.length];
      return rectangleEdges.every(
        ([rectangleStart, rectangleEnd]) =>
          !segmentsProperlyIntersect(
            rectangleStart,
            rectangleEnd,
            start,
            end,
          ),
      );
    }),
  );
}

function getLabelCenters(polygon) {
  const bounds = getRingBounds(polygon[0]);
  const centers = [];
  for (let row = 0; row < LABEL_GRID_SIZE; row += 1) {
    for (let column = 0; column < LABEL_GRID_SIZE; column += 1) {
      const point = [
        bounds[0] +
          ((column + 0.5) * (bounds[2] - bounds[0])) / LABEL_GRID_SIZE,
        bounds[1] +
          ((row + 0.5) * (bounds[3] - bounds[1])) / LABEL_GRID_SIZE,
      ];
      if (isPointInPolygon(point, polygon)) {
        centers.push(point);
      }
    }
  }
  return centers;
}

function getBestLabelFit(name, polygons) {
  const widthAtUnitSize =
    Math.max(name.length, 1) *
    MAP_LABEL_CHARACTER_WIDTH_RATIO *
    (1 + MAP_LABEL_MARGIN_RATIO);
  const heightAtUnitSize = 1 + MAP_LABEL_MARGIN_RATIO;
  let best = null;
  for (const polygon of polygons) {
    const bounds = getRingBounds(polygon[0]);
    const maximumSize = Math.max(bounds[2] - bounds[0], bounds[3] - bounds[1]);
    for (const center of getLabelCenters(polygon)) {
      for (const angle of LABEL_ANGLES) {
        let minimum = 0;
        let maximum = maximumSize;
        for (let iteration = 0; iteration < LABEL_FIT_ITERATIONS; iteration += 1) {
          const fontSize = (minimum + maximum) / 2;
          const rectangle = getRectangle(
            center,
            fontSize * widthAtUnitSize,
            fontSize * heightAtUnitSize,
            angle,
          );
          if (doesRectangleFit(rectangle, polygon)) {
            minimum = fontSize;
          } else {
            maximum = fontSize;
          }
        }
        if (!best || minimum > best.fontSize) {
          best = { angle, fontSize: minimum, position: center };
        }
      }
    }
  }
  return best;
}

export function buildRegionLabels(features, projection) {
  if (features.length > MAP_MAX_LABEL_COUNT) {
    return [];
  }
  return features
    .map((feature) => {
      const fit = getBestLabelFit(
        feature.properties.name,
        getProjectedPolygons(feature, projection),
      );
      if (!fit) {
        return null;
      }
      return {
        ...fit,
        backgroundColor: feature.fill ?? MAP_UNKNOWN_COLOR,
        id: feature.id,
        name: feature.properties.name,
      };
    })
    .filter(
      (label) =>
        label &&
        label.position.every(Number.isFinite) &&
        Number.isFinite(label.fontSize),
    );
}
