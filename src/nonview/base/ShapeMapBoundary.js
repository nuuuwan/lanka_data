import { getHexPoints, getSquarePoints } from "./ShapeMapPoints.js";

function getEdgeKey(start, end) {
  const pointKey = ([x, y]) => `${x.toFixed(6)},${y.toFixed(6)}`;
  return [pointKey(start), pointKey(end)].sort().join(":");
}

function getBoundaryEdges(shapes, getPoints) {
  const edgeGroups = new Map();
  for (const { id, center } of shapes) {
    const points = getPoints(center);
    for (let index = 0; index < points.length; index += 1) {
      const start = points[index];
      const end = points[(index + 1) % points.length];
      const key = getEdgeKey(start, end);
      const edges = edgeGroups.get(key) ?? [];
      edges.push({ id, start, end });
      edgeGroups.set(key, edges);
    }
  }
  return [...edgeGroups.values()]
    .filter(
      (edges) =>
        edges.length === 1 || edges.some(({ id }) => id !== edges[0].id),
    )
    .map(([edge]) => edge);
}

export function getHexBoundaryEdges(shapes, radius) {
  return getBoundaryEdges(shapes, (center) => getHexPoints(center, radius));
}

export function getSquareBoundaryEdges(shapes, size) {
  return getBoundaryEdges(shapes, (center) => getSquarePoints(center, size));
}
