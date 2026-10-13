export function getColorForDimension(id, data) {
  const colorKey = `${id}Color`;
  return data.find((row) => row[colorKey])?.[colorKey];
}

export function getDimensionKeys(data) {
  const rows = Array.isArray(data) ? data : [data];
  const totals = {};
  for (const row of rows) {
    if (!row) continue;
    for (const key of Object.keys(row)) {
      if (key !== "id" && key !== "_barWidth" && !key.endsWith("Color")) {
        totals[key] = (totals[key] || 0) + (row[key] || 0);
      }
    }
  }
  return Object.keys(totals).sort(
    (a, b) => totals[b] - totals[a] || a.localeCompare(b),
  );
}

export function getDimensions(data) {
  return getDimensionKeys(data).map((key) => ({ id: key, value: key }));
}

export function sortByDominantCategory(data) {
  if (!data.length) return data;
  const keys = getDimensionKeys(data);
  const dominant = keys[0];
  if (!dominant) return data;
  const percentage = (row) => {
    const total = keys.reduce((sum, key) => sum + (row[key] || 0), 0);
    return total ? (row[dominant] || 0) / total : 0;
  };
  return [...data].sort((a, b) => percentage(b) - percentage(a));
}
