import { geoMercator } from "d3-geo";

import {
  MAP_HEIGHT,
  MAP_LABEL_CHARACTER_WIDTH_RATIO,
  MAP_LABEL_MARGIN_RATIO,
  MAP_PADDING,
  MAP_WIDTH,
} from "../../constants/MAP.js";

export function getGeoCoordinates(features) {
  const coordinates = [];
  function collect(value) {
    if (!Array.isArray(value)) return;
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
  const [x, y] = projection.translate();
  return {
    projection,
    projectionScale: projection.scale(),
    projectionTranslation: [x / MAP_WIDTH, y / MAP_HEIGHT],
  };
}

export function getFittedLabelFontSize(label, width, height) {
  const unitWidth = Math.max(label.length, 1) * MAP_LABEL_CHARACTER_WIDTH_RATIO;
  const padding = 1 + MAP_LABEL_MARGIN_RATIO;
  return Math.min(height / padding, width / (unitWidth * padding));
}
