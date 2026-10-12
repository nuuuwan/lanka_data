import { getMarkColor } from "../constants/COLORS.js";

export function getColorForKey(key, data) {
  return data.find((row) => row[`${key}Color`])?.[`${key}Color`];
}

export function getStackKeys(data) {
  return Object.keys(data[0] ?? {}).filter(
    (key) => key !== "id" && key !== "_barWidth" && !key.endsWith("Color"),
  );
}

export function getStackLegendItems(keys, data) {
  return keys.map((key) => ({
    id: key,
    label: key,
    color: getMarkColor(getColorForKey(key, data)),
  }));
}
