export function getColorForDimension(id, data) {
  const colorKey = `${id}Color`;
  return data.find((row) => row[colorKey])?.[colorKey];
}

export function getDimensionKeys(row) {
  return Object.keys(row).filter(
    (key) => key !== "id" && key !== "_barWidth" && !key.endsWith("Color"),
  );
}

export function getDimensions(data) {
  return data.length
    ? getDimensionKeys(data[0]).map((key) => ({ id: key, value: key }))
    : [];
}

export function sortByDominantCategory(data) {
  if (!data.length) return data;
  const keys = getDimensionKeys(data[0]);
  const totals = Object.fromEntries(keys.map((key) => [key, 0]));
  data.forEach((row) =>
    keys.forEach((key) => {
      totals[key] += row[key] || 0;
    }),
  );
  const dominant = keys.reduce((best, key) =>
    totals[key] > totals[best] ? key : best,
  );
  const percentage = (row) => {
    const total = keys.reduce((sum, key) => sum + (row[key] || 0), 0);
    return total ? (row[dominant] || 0) / total : 0;
  };
  return [...data].sort((a, b) => percentage(b) - percentage(a));
}
