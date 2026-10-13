import { getMarkColor } from "../constants/COLORS.js";

export function getColorForKey(key, data) {
  return data.find((row) => row[`${key}Color`])?.[`${key}Color`];
}

export function getStackKeys(data) {
  const totals = {};
  for (const row of data) {
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

export function getStackLegendItems(keys, data) {
  return keys.map((key) => ({
    id: key,
    label: key,
    color: getMarkColor(getColorForKey(key, data)),
  }));
}
