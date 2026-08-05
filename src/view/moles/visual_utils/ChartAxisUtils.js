export const X_AXIS_BOTTOM_MARGIN = 220;
export const X_AXIS_LABEL_ROTATION = -90;
export const X_AXIS_LEGEND_OFFSET = 200;
export const BAR_CHART_MIN_HEIGHT_PX = 580;

const X_AXIS_MAX_TICK_FONT_SIZE = 14;
const X_AXIS_MIN_TICK_FONT_SIZE = 10;
const X_AXIS_TICK_FONT_BUDGET = 168;

export function getDenseAxisTheme(theme, tickCount) {
  const fontSize = Math.max(
    X_AXIS_MIN_TICK_FONT_SIZE,
    Math.min(X_AXIS_MAX_TICK_FONT_SIZE, X_AXIS_TICK_FONT_BUDGET / tickCount),
  );
  return {
    ...theme,
    axis: {
      ...theme.axis,
      ticks: { text: { ...theme.axis?.ticks?.text, fontSize } },
    },
  };
}

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
