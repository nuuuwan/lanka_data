function getStackKeys(data) {
  return [
    ...new Set(
      data.flatMap((row) =>
        Object.keys(row).filter(
          (key) =>
            key !== "id" && key !== "_barWidth" && !key.endsWith("Color"),
        ),
      ),
    ),
  ];
}

export function getLargestStackKey(data) {
  const totalByStackKey = new Map();
  for (const row of data) {
    for (const stackKey of getStackKeys([row])) {
      totalByStackKey.set(
        stackKey,
        (totalByStackKey.get(stackKey) || 0) + (row[stackKey] || 0),
      );
    }
  }
  return Array.from(totalByStackKey.entries()).reduce(
    (largestKey, [stackKey, total]) =>
      total > totalByStackKey.get(largestKey) ? stackKey : largestKey,
    totalByStackKey.keys().next().value,
  );
}

export function sortByStackValue(data, stackKey) {
  if (!stackKey) {
    return data;
  }
  return [...data].sort((left, right) => {
    const valueComparison = (right[stackKey] || 0) - (left[stackKey] || 0);
    return (
      valueComparison ||
      String(left.id).localeCompare(String(right.id), undefined, {
        numeric: true,
      })
    );
  });
}
