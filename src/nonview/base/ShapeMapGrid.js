const GRID_FACTOR = 1.3;
const MAX_GRID_ITERATIONS = 12;
const MAX_SHAPE_ERROR = 0.1;
const HEX_AREA_FACTOR = (3 * Math.sqrt(3)) / 2;
const SQUARE_AREA_FACTOR = 1;

function roundHalfEven(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (Math.abs(fraction - 0.5) > Number.EPSILON * Math.abs(value)) {
    return Math.round(value);
  }
  return floor % 2 === 0 ? floor : floor + 1;
}

function regionError(actual, ideal) {
  return Math.abs(actual - ideal) / ideal;
}

function maxError(weights, valuePerShape) {
  return Math.max(
    ...weights.map((weight) => {
      const ideal = weight / valuePerShape;
      return regionError(Math.max(1, roundHalfEven(ideal)), ideal);
    }),
  );
}

function getCandidates(weights) {
  const nMax = Math.floor(0.5 / MAX_SHAPE_ERROR) + 2;
  const minimum = Math.min(...weights);
  const cap = minimum * (1 + MAX_SHAPE_ERROR);
  const candidates = new Set([minimum * 2 * MAX_SHAPE_ERROR]);
  for (const weight of weights) {
    for (let count = 1; count <= nMax; count += 1) {
      const value = (weight * (1 + MAX_SHAPE_ERROR)) / count;
      if (value <= cap) {
        candidates.add(value);
      }
    }
  }
  return [...candidates].sort((a, b) => b - a);
}

export function getValuePerShape(weights, maxTotalCount = Infinity) {
  const positiveWeights = weights.filter((weight) => weight > 0);
  if (!positiveWeights.length) {
    return null;
  }
  const tolerance = MAX_SHAPE_ERROR + 1e-9;
  const valuePerShape =
    getCandidates(positiveWeights).find(
      (candidate) => maxError(positiveWeights, candidate) <= tolerance,
    ) ?? Math.min(...positiveWeights) * 2 * MAX_SHAPE_ERROR;
  if (!Number.isFinite(maxTotalCount)) {
    return valuePerShape;
  }

  const targetCount = Math.max(maxTotalCount, weights.length);
  const getTotalCount = (candidate) =>
    weights.reduce(
      (total, weight) => total + Math.max(1, roundHalfEven(weight / candidate)),
      0,
    );
  if (getTotalCount(valuePerShape) <= targetCount) {
    return valuePerShape;
  }

  let lower = valuePerShape;
  let upper = Math.max(...positiveWeights);
  for (let iteration = 0; iteration < 64; iteration += 1) {
    const middle = (lower + upper) / 2;
    if (getTotalCount(middle) > targetCount) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return upper;
}

export function getShapeCounts(regionToWeight, valuePerShape = null) {
  const entries = Object.entries(regionToWeight);
  const resolvedValuePerShape =
    valuePerShape ?? getValuePerShape(entries.map(([, weight]) => weight));
  return Object.fromEntries(
    entries.map(([regionId, weight]) => [
      regionId,
      resolvedValuePerShape
        ? Math.max(1, roundHalfEven(weight / resolvedValuePerShape))
        : 1,
    ]),
  );
}

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

export function buildHexGrid(bounds, totalCount) {
  const { centers, size } = buildGrid(
    bounds,
    totalCount,
    HEX_AREA_FACTOR,
    getHexCenters,
  );
  return { centers, radius: size };
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

export function buildSquareGrid(bounds, totalCount) {
  return buildGrid(bounds, totalCount, SQUARE_AREA_FACTOR, getSquareCenters);
}
