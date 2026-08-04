const MAX_SHAPE_ERROR = 0.1;

function roundHalfEven(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (Math.abs(fraction - 0.5) > Number.EPSILON * Math.abs(value)) {
    return Math.round(value);
  }
  return floor % 2 === 0 ? floor : floor + 1;
}

function maxError(weights, valuePerShape) {
  return Math.max(
    ...weights.map((weight) => {
      const ideal = weight / valuePerShape;
      return Math.abs(Math.max(1, roundHalfEven(ideal)) - ideal) / ideal;
    }),
  );
}

function getCandidates(weights) {
  const maxCount = Math.floor(0.5 / MAX_SHAPE_ERROR) + 2;
  const minimum = Math.min(...weights);
  const cap = minimum * (1 + MAX_SHAPE_ERROR);
  const candidates = new Set([minimum * 2 * MAX_SHAPE_ERROR]);
  for (const weight of weights) {
    for (let count = 1; count <= maxCount; count += 1) {
      const value = (weight * (1 + MAX_SHAPE_ERROR)) / count;
      if (value <= cap) {
        candidates.add(value);
      }
    }
  }
  return [...candidates].sort((left, right) => right - left);
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
