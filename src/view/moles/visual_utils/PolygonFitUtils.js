export function getProjectedPolygons(feature, projection) {
  const projectRing = (ring) => ring.map(projection);
  if (feature.geometry.type === "Polygon")
    return [feature.geometry.coordinates.map(projectRing)];
  if (feature.geometry.type === "MultiPolygon")
    return feature.geometry.coordinates.map((polygon) =>
      polygon.map(projectRing),
    );
  return [];
}

export function getRingBounds(ring) {
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
  const cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax);
  return (
    Math.abs(cross) <= 1e-7 &&
    px >= Math.min(ax, bx) - 1e-7 &&
    px <= Math.max(ax, bx) + 1e-7 &&
    py >= Math.min(ay, by) - 1e-7 &&
    py <= Math.max(ay, by) + 1e-7
  );
}

function isPointInRing(point, ring) {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; index++) {
    const start = ring[previous];
    const end = ring[index];
    if (isPointOnSegment(point, start, end)) return true;
    if (
      (start[1] > point[1]) !== (end[1] > point[1]) &&
      point[0] <
        ((end[0] - start[0]) * (point[1] - start[1])) / (end[1] - start[1]) +
          start[0]
    )
      inside = !inside;
    previous = index;
  }
  return inside;
}

export function isPointInPolygon(point, polygon) {
  return (
    isPointInRing(point, polygon[0]) &&
    polygon.slice(1).every((hole) => !isPointInRing(point, hole))
  );
}

function crossProduct([ax, ay], [bx, by], [cx, cy]) {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

export function doesRectangleFit(rectangle, polygon) {
  if (!rectangle.every((point) => isPointInPolygon(point, polygon)))
    return false;
  const rectangleEdges = rectangle.map((start, index) => [
    start,
    rectangle[(index + 1) % rectangle.length],
  ]);
  return polygon.every((ring) =>
    ring.every((start, index) => {
      const end = ring[(index + 1) % ring.length];
      return rectangleEdges.every(([a, b]) => {
        const products = [
          crossProduct(a, b, start) * crossProduct(a, b, end),
          crossProduct(start, end, a) * crossProduct(start, end, b),
        ];
        return !(products[0] < -1e-7 && products[1] < -1e-7);
      });
    }),
  );
}
