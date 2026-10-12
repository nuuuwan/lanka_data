import { VISUAL_LABEL_MIN_FONT_SIZE } from "../../../nonview/constants/ChartLabels.js";
import { X_AXIS_LABEL_ROTATION } from "../visual_utils/ChartAxisUtils.js";

export function BarLabelsLayer({ data }) {
  return (
    <>
      {data.map((datum) => {
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
              fontSize: VISUAL_LABEL_MIN_FONT_SIZE,
              fill: "#333",
            }}
          >
            {datum.id}
          </text>
        );
      })}
    </>
  );
}
