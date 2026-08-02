const GRID_FACTOR = 1.3;
const MAX_GRID_ITERATIONS = 12;
const MAX_SHAPE_ERROR = 0.1;
const HEX_AREA_FACTOR = (3 * Math.sqrt(3)) / 2;

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

export function getValuePerShape(weights) {
  const positiveWeights = weights.filter((weight) => weight > 0);
  if (!positiveWeights.length) {
    return null;
  }
  const tolerance = MAX_SHAPE_ERROR + 1e-9;
  return (
    getCandidates(positiveWeights).find(
      (candidate) => maxError(positiveWeights, candidate) <= tolerance,
    ) ?? Math.min(...positiveWeights) * 2 * MAX_SHAPE_ERROR
  );
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
  const [minX, minY, maxX, maxY] = bounds;
  const target = Math.max(totalCount * GRID_FACTOR, totalCount + 1);
  const area = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  let radius = Math.sqrt(area / (Math.max(target, 1) * HEX_AREA_FACTOR));
  let centers = getHexCenters(bounds, radius);
  for (
    let iteration = 0;
    iteration < MAX_GRID_ITERATIONS && centers.length < totalCount;
    iteration += 1
  ) {
    radius *= 0.85;
    centers = getHexCenters(bounds, radius);
  }
  return { centers, radius };
}

function solveAssignment(cost) {
  const rowCount = cost.length;
  const columnCount = cost[0]?.length ?? 0;
  if (!rowCount || !columnCount) {
    return [];
  }
  if (rowCount > columnCount) {
    throw new Error("Shape assignment requires at least one center per slot.");
  }

  const rowPotential = Array(rowCount + 1).fill(0);
  const columnPotential = Array(columnCount + 1).fill(0);
  const matchedRow = Array(columnCount + 1).fill(0);
  const previousColumn = Array(columnCount + 1).fill(0);

  for (let row = 1; row <= rowCount; row += 1) {
    matchedRow[0] = row;
    let currentColumn = 0;
    const minimum = Array(columnCount + 1).fill(Infinity);
    const used = Array(columnCount + 1).fill(false);
    do {
      used[currentColumn] = true;
      const currentRow = matchedRow[currentColumn];
      let delta = Infinity;
      let nextColumn = 0;
      for (let column = 1; column <= columnCount; column += 1) {
        if (used[column]) {
          continue;
        }
        const reducedCost =
          cost[currentRow - 1][column - 1] -
          rowPotential[currentRow] -
          columnPotential[column];
        if (reducedCost < minimum[column]) {
          minimum[column] = reducedCost;
          previousColumn[column] = currentColumn;
        }
        if (minimum[column] < delta) {
          delta = minimum[column];
          nextColumn = column;
        }
      }
      for (let column = 0; column <= columnCount; column += 1) {
        if (used[column]) {
          rowPotential[matchedRow[column]] += delta;
          columnPotential[column] -= delta;
        } else {
          minimum[column] -= delta;
        }
      }
      currentColumn = nextColumn;
    } while (matchedRow[currentColumn] !== 0);

    do {
      const nextColumn = previousColumn[currentColumn];
      matchedRow[currentColumn] = matchedRow[nextColumn];
      currentColumn = nextColumn;
    } while (currentColumn !== 0);
  }

  const assignment = Array(rowCount);
  for (let column = 1; column <= columnCount; column += 1) {
    if (matchedRow[column]) {
      assignment[matchedRow[column] - 1] = column - 1;
    }
  }
  return assignment;
}

export function assignShapes(regions, centers) {
  const slots = regions.flatMap(({ id, centroid, count }) =>
    Array.from({ length: count }, () => ({ id, centroid })),
  );
  const cost = slots.map(({ centroid: [cx, cy] }) =>
    centers.map(([x, y]) => (cx - x) ** 2 + (cy - y) ** 2),
  );
  return solveAssignment(cost).map((centerIndex, slotIndex) => ({
    id: slots[slotIndex].id,
    center: centers[centerIndex],
  }));
}

export function getHexPoints([x, y], radius) {
  return Array.from({ length: 6 }, (_, pointIndex) => {
    const angle = Math.PI / 2 + (Math.PI / 3) * pointIndex;
    return [x + radius * Math.cos(angle), y + radius * Math.sin(angle)];
  });
}
