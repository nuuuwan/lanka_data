import StringUtils from "../../../nonview/base/String.js";
import { X_AXIS_LABEL_ROTATION } from "../visual_utils/ChartAxisUtils.js";

const getFontScale = (screenWidth) => screenWidth / 1200;

function getAxisLabel(value, width) {
  if (width >= 20) {
    return value;
  }
  if (width >= 14) {
    return StringUtils.shorten(value, 3);
  }
  if (width >= 9) {
    return StringUtils.shorten(value, 2);
  }
  return StringUtils.shorten(value, 1);
}

export function BarLabelsLayer({ data, screenWidth }) {
  const congestionScale = Math.max(0.7, 1 / Math.sqrt(data.length));
  const fontScale = getFontScale(screenWidth) * congestionScale;
  return (
    <>
      {data.map((datum) => {
        const label = getAxisLabel(datum.id, datum.width);
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
              fontSize: Math.max(
                6,
                Math.min(10, (Math.max(datum.width, 8) / 10) * fontScale),
              ),
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
