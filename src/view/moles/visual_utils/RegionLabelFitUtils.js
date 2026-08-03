import {
  MAP_LABEL_CHARACTER_WIDTH_RATIO,
  MAP_LABEL_MARGIN_RATIO,
} from "../../_cons/MapCons.js";
import {
  doesRectangleFit,
  getProjectedPolygons,
  getRingBounds,
  isPointInPolygon,
} from "./PolygonFitUtils.js";

const ANGLES = [0, 30, 60, 90, 120, 150];
const GRID_SIZE = 8;
const ITERATIONS = 10;

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

function getLabelCenters(polygon) {
  const bounds = getRingBounds(polygon[0]);
  const centers = [];
  for (let row = 0; row < GRID_SIZE; row++) {
    for (let column = 0; column < GRID_SIZE; column++) {
      const point = [
        bounds[0] + ((column + 0.5) * (bounds[2] - bounds[0])) / GRID_SIZE,
        bounds[1] + ((row + 0.5) * (bounds[3] - bounds[1])) / GRID_SIZE,
      ];
      if (isPointInPolygon(point, polygon)) centers.push(point);
    }
  }
  return centers;
}

export function getBestLabelFit(name, feature, projection) {
  const width =
    Math.max(name.length, 1) *
    MAP_LABEL_CHARACTER_WIDTH_RATIO *
    (1 + MAP_LABEL_MARGIN_RATIO);
  const height = 1 + MAP_LABEL_MARGIN_RATIO;
  let best = null;
  for (const polygon of getProjectedPolygons(feature, projection)) {
    const bounds = getRingBounds(polygon[0]);
    const maximum = Math.max(bounds[2] - bounds[0], bounds[3] - bounds[1]);
    for (const center of getLabelCenters(polygon)) {
      for (const angle of ANGLES) {
        let low = 0;
        let high = maximum;
        for (let iteration = 0; iteration < ITERATIONS; iteration++) {
          const size = (low + high) / 2;
          if (
            doesRectangleFit(
              getRectangle(center, size * width, size * height, angle),
              polygon,
            )
          )
            low = size;
          else high = size;
        }
        if (!best || low > best.fontSize)
          best = { angle, fontSize: low, position: center };
      }
    }
  }
  return best;
}
