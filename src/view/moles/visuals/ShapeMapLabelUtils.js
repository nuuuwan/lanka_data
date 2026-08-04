import StringUtils from "../../../nonview/base/String.js";
import { MAP_LABEL_MIN_FONT_SIZE } from "../../_cons/MapCons.js";
import { getFittedLabelFontSize } from "../visual_utils/GeoVisualUtils.js";

function getDisplayLabel(name, width, height) {
  const candidates = [
    name,
    StringUtils.shorten(name, 3),
    StringUtils.shorten(name, 2),
    StringUtils.shorten(name, 1),
  ];
  let bestName = name;
  let bestFontSize = getFittedLabelFontSize(name, width, height);
  if (bestFontSize >= MAP_LABEL_MIN_FONT_SIZE) {
    return { name: bestName, fontSize: bestFontSize };
  }
  for (const candidate of candidates) {
    const fontSize = getFittedLabelFontSize(candidate, width, height);
    if (fontSize > bestFontSize) {
      bestName = candidate;
      bestFontSize = fontSize;
    }
  }
  return { name: bestName, fontSize: bestFontSize };
}

export function getShapeMapLabels(
  shapes,
  regionById,
  shapeSize,
  getBestLabelFit,
) {
  const centersById = new Map();
  for (const { id, center } of shapes) {
    const centers = centersById.get(id) ?? [];
    centers.push(center);
    centersById.set(id, centers);
  }
  return [...centersById].map(([id, centers]) => {
    const region = regionById.get(id);
    const fullName = region.feature.properties.name;
    const label = {
      ...getBestLabelFit(centers, shapeSize),
      color: region.display.color,
      id,
      name: fullName,
    };
    const displayLabel = getDisplayLabel(label.name, label.width, label.height);
    return {
      ...label,
      fontSize: displayLabel.fontSize,
      name: displayLabel.name,
    };
  });
}

export function getShapeMapViewBox(shapes, extent) {
  const xValues = shapes.map(({ center }) => center[0]);
  const yValues = shapes.map(({ center }) => center[1]);
  const minX = Math.min(...xValues) - extent;
  const minY = Math.min(...yValues) - extent;
  return [
    minX,
    minY,
    Math.max(...xValues) + extent - minX,
    Math.max(...yValues) + extent - minY,
  ];
}
