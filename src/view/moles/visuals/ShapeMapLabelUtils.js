import { getFittedLabelFontSize } from "../visual_utils/GeoVisualUtils.js";

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
    const label = {
      ...getBestLabelFit(centers, shapeSize),
      color: region.display.color,
      id,
      name: region.feature.properties.name,
    };
    return {
      ...label,
      fontSize: getFittedLabelFontSize(label.name, label.width, label.height),
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
