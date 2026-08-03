import { FONT_FAMILY } from "../../../AppTheme.js";
import getLargestLabelFit from "../../../nonview/base/getLargestLabelFit.js";
import { IN_BAR_LABEL_STROKE_WIDTH } from "../../../nonview/constants/ChartLabels.js";
import FormatUtils from "../visual_utils/FormatUtils.js";

function getLabel(bar) {
  const value =
    bar.data?.value ?? bar.datum?.data?.[bar.dimension?.id] ?? bar.value;
  return FormatUtils.humanizeValue(value);
}

export default function InBarLabels({ bars }) {
  return (
    <g aria-hidden="true" pointerEvents="none">
      {bars.map((bar) => {
        const label = getLabel(bar);
        const fit = getLargestLabelFit(label, bar.width, bar.height);
        if (!fit) return null;

        const light = FormatUtils.isLightColor(bar.color);
        const x = bar.x + bar.width / 2;
        const y = bar.y + bar.height / 2;
        return (
          <text
            key={bar.key}
            x={x}
            y={y}
            textAnchor="middle"
            dominantBaseline="central"
            transform={`rotate(${fit.rotation} ${x} ${y})`}
            style={{
              fontFamily: FONT_FAMILY,
              fontSize: fit.fontSize,
              fill: light ? "black" : "white",
              stroke: light ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)",
              strokeWidth: IN_BAR_LABEL_STROKE_WIDTH,
            }}
          >
            {label}
          </text>
        );
      })}
    </g>
  );
}
