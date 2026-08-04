const GRID_FACTOR = 1.3;
const MAX_GRID_ITERATIONS = 12;
const HEX_AREA_FACTOR = (3 * Math.sqrt(3)) / 2;

function getHexCenters(bounds, radius) {
  const [minX, minY, maxX, maxY] = bounds;
  const dx = Math.sqrt(3) * radius;
  const dy = 1.5 * radius;
  const centers = [];
  for (let row = 0, y = minY; y <= maxY + dy; row += 1, y += dy) {
    for (let x = minX + (row % 2) * (dx / 2); x <= maxX + dx; x += dx) {
      centers.push([x, y]);
    }
  }
  return centers;
}

function getSquareCenters(bounds, size) {
  const [minX, minY, maxX, maxY] = bounds;
  const centers = [];
  for (let y = minY + size / 2; y <= maxY + size; y += size) {
    for (let x = minX + size / 2; x <= maxX + size; x += size) {
      centers.push([x, y]);
    }
  }
  return centers;
}

function buildGrid(bounds, totalCount, areaFactor, getCenters) {
  const [minX, minY, maxX, maxY] = bounds;
  const target = Math.max(totalCount * GRID_FACTOR, totalCount + 1);
  const area = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  let size = Math.sqrt(area / (Math.max(target, 1) * areaFactor));
  let centers = getCenters(bounds, size);
  for (
    let iteration = 0;
    iteration < MAX_GRID_ITERATIONS && centers.length < totalCount;
    iteration += 1
  ) {
    size *= 0.85;
    centers = getCenters(bounds, size);
  }
  return { centers, size };
}

export function buildHexGrid(bounds, totalCount) {
  const { centers, size } = buildGrid(
    bounds,
    totalCount,
    HEX_AREA_FACTOR,
    getHexCenters,
  );
  return { centers, radius: size };
}

export function buildSquareGrid(bounds, totalCount) {
  return buildGrid(bounds, totalCount, 1, getSquareCenters);
}
