import {
  IN_BAR_LABEL_CHARACTER_WIDTH_RATIO,
  IN_BAR_LABEL_MIN_FONT_SIZE,
  IN_BAR_LABEL_PADDING,
} from "../constants/ChartLabels.js";

export default function getLargestLabelFit(label, width, height) {
  const availableWidth = width - IN_BAR_LABEL_PADDING * 2;
  const availableHeight = height - IN_BAR_LABEL_PADDING * 2;
  if (availableWidth <= 0 || availableHeight <= 0) return null;

  const textWidth = Math.max(String(label).length, 1) *
    IN_BAR_LABEL_CHARACTER_WIDTH_RATIO;
  const horizontalFontSize = Math.min(
    availableHeight,
    availableWidth / textWidth,
  );
  const verticalFontSize = Math.min(
    availableWidth,
    availableHeight / textWidth,
  );
  const isVertical = verticalFontSize > horizontalFontSize;
  const fontSize = isVertical ? verticalFontSize : horizontalFontSize;

  return fontSize >= IN_BAR_LABEL_MIN_FONT_SIZE
    ? { fontSize, rotation: isVertical ? -90 : 0 }
    : null;
}
