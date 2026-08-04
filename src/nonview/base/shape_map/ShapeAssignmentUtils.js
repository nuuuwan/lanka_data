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
