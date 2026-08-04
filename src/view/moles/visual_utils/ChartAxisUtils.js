export const X_AXIS_LABEL_ROTATION = -45;
export const X_AXIS_LEGEND_OFFSET = 80;

export function getSlantedXAxis(xAxisLabel, overrides = {}) {
  return {
    tickSize: 5,
    tickPadding: 5,
    tickRotation: X_AXIS_LABEL_ROTATION,
    legend: xAxisLabel,
    legendPosition: "middle",
    legendOffset: X_AXIS_LEGEND_OFFSET,
    ...overrides,
  };
}
