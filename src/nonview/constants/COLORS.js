export const SEMANTIC_PALETTE = Object.freeze({
  neutral: "#9e9e9e",
  positive: "#6CD038",
  negative: "#D05D38",
  information: "#3840D0",
  warning: "#D0AF38",
  accent: "#8238D0",
});

export function getMarkColor(color) {
  return color ?? SEMANTIC_PALETTE.neutral;
}
