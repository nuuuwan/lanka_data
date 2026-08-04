import StringUtils from "../../../nonview/base/String.js";
import { X_AXIS_LABEL_ROTATION } from "../visual_utils/ChartAxisUtils.js";

const getFontScale = (screenWidth) => screenWidth / 1200;
const CHARACTER_WIDTH_RATIO = 0.6;
const LABEL_ROTATION_RADIANS =
  (Math.abs(X_AXIS_LABEL_ROTATION) * Math.PI) / 180;

function getAxisLabel(value, width, fontSize) {
  const availableWidth = width / Math.cos(LABEL_ROTATION_RADIANS);
  for (const candidate of [
    value,
    StringUtils.shorten(value, 3),
    StringUtils.shorten(value, 2),
    StringUtils.shorten(value, 1),
  ]) {
    const textWidth = candidate.length * fontSize * CHARACTER_WIDTH_RATIO;
    if (textWidth <= availableWidth) {
      return candidate;
    }
  }
  return StringUtils.shorten(value, 1);
}

export function BarLabelsLayer({ data, screenWidth }) {
  const congestionScale = Math.max(0.7, 1 / Math.sqrt(data.length));
  const fontScale = getFontScale(screenWidth) * congestionScale;
  return (
    <>
      {data.map((datum) => {
        const fontSize = Math.max(
          10,
          Math.min(14, (Math.max(datum.width, 10) / 10) * fontScale),
        );
        const label = getAxisLabel(datum.id, datum.width, fontSize);
        return (
          <text
            key={datum.id}
            x={datum.x + datum.width / 2}
            y={datum.y + datum.height + 8}
            textAnchor="end"
            dominantBaseline="hanging"
            transform={`rotate(${X_AXIS_LABEL_ROTATION} ${
              datum.x + datum.width / 2
            } ${datum.y + datum.height + 8})`}
            style={{
              fontSize,
              fill: "#333",
            }}
          >
            {label}
          </text>
        );
      })}
    </>
  );
}
