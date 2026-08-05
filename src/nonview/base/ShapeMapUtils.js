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

  const getTotalCount = (candidate) =>
    weights.reduce(
      (total, weight) => total + roundHalfEven(weight / candidate),
      0,
    );
  if (getTotalCount(valuePerShape) <= maxTotalCount) {
    return valuePerShape;
  }

  let lower = valuePerShape;
  let upper = Math.max(...positiveWeights);
  for (let iteration = 0; iteration < 64; iteration += 1) {
    const middle = (lower + upper) / 2;
    if (getTotalCount(middle) > maxTotalCount) {
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
      resolvedValuePerShape ? roundHalfEven(weight / resolvedValuePerShape) : 1,
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

export function getSquarePoints([x, y], size) {
  const halfSize = size / 2;
  return [
    [x - halfSize, y - halfSize],
    [x + halfSize, y - halfSize],
    [x + halfSize, y + halfSize],
    [x - halfSize, y + halfSize],
  ];
}

function getAxes(angleDegrees) {
  const angle = (angleDegrees * Math.PI) / 180;
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [
    [cosine, sine],
    [-sine, cosine],
  ];
}

function projectPoint(point, axis) {
  return point[0] * axis[0] + point[1] * axis[1];
}

function getLongestRun(items, step) {
  const sortedItems = [...items].sort((a, b) => a[0] - b[0]);
  let run = [sortedItems[0]];
  let longestRun = run;
  const tolerance = step * 0.5;
  for (let index = 1; index < sortedItems.length; index += 1) {
    const previous = sortedItems[index - 1];
    const current = sortedItems[index];
    run =
      Math.abs(current[0] - previous[0] - step) < tolerance
        ? [...run, current]
        : [current];
    if (run.length > longestRun.length) {
      longestRun = run;
    }
  }
  return longestRun;
}

export function getBestHexLabelFit(points, radius) {
  const step = Math.sqrt(3) * radius;
  const lineSpacing = 1.5 * radius;
  let best = null;
  for (const angle of [0, 60, -60]) {
    const [horizontalAxis, verticalAxis] = getAxes(angle);
    const lines = new Map();
    for (const point of points) {
      const key = Math.round(projectPoint(point, verticalAxis) / lineSpacing);
      const line = lines.get(key) ?? [];
      line.push([projectPoint(point, horizontalAxis), point]);
      lines.set(key, line);
    }

    for (const line of lines.values()) {
      const run = getLongestRun(line, step);
      if (!best || run.length > best.run.length) {
        best = { angle, run };
      }
    }
  }
  const first = best.run[0][1];
  const last = best.run.at(-1)[1];
  return {
    center: [(first[0] + last[0]) / 2, (first[1] + last[1]) / 2],
    width: best.run.length * step,
    height: lineSpacing,
    angle: best.angle,
  };
}

export function getBestSquareLabelFit(points, size) {
  let best = null;
  for (const angle of [0, 90]) {
    const [horizontalAxis, verticalAxis] = getAxes(angle);
    const lines = new Map();
    for (const point of points) {
      const key = Math.round(projectPoint(point, verticalAxis) / size);
      const line = lines.get(key) ?? [];
      line.push([projectPoint(point, horizontalAxis), point]);
      lines.set(key, line);
    }
    for (const line of lines.values()) {
      const run = getLongestRun(line, size);
      if (!best || run.length > best.run.length) {
        best = { angle, run };
      }
    }
  }
  const first = best.run[0][1];
  const last = best.run.at(-1)[1];
  return {
    center: [(first[0] + last[0]) / 2, (first[1] + last[1]) / 2],
    width: best.run.length * size,
    height: size,
    angle: best.angle,
  };
}

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
