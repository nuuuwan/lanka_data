import { animated } from "@react-spring/web";

import FormatUtils from "../visual_utils/FormatUtils.js";
import {
  VISUAL_LABEL_MIN_FONT_SIZE,
  PIE_LABEL_CHARACTER_WIDTH_RATIO,
  PIE_LABEL_PADDING_RATIO,
  PIE_LABEL_MAX_FONT_SIZE,
} from "../../../nonview/constants/ChartLabels.js";

function getFontSize(
  { endAngle, innerRadius, outerRadius, startAngle },
  label,
) {
  const middleRadius = (innerRadius + outerRadius) / 2;
  const arcLength = middleRadius * Math.abs(endAngle - startAngle);
  const radialSpace = outerRadius - innerRadius;
  const textWidth = Math.max(label.length, 1) * PIE_LABEL_CHARACTER_WIDTH_RATIO;
  const fit = Math.min(radialSpace, arcLength / textWidth);
  return Math.min(
    PIE_LABEL_MAX_FONT_SIZE,
    Math.max(VISUAL_LABEL_MIN_FONT_SIZE, fit * PIE_LABEL_PADDING_RATIO),
  );
}

export default function PieArcLabel({ datum, label, style }) {
  const color = FormatUtils.isLightColor(datum.color) ? "#000000" : "#ffffff";
  const fontSize = getFontSize(datum.arc, label);

  return (
    <animated.text
      dominantBaseline="central"
      textAnchor="middle"
      transform={style.transform}
      fill={color}
      style={{ fontSize }}
    >
      {label}
    </animated.text>
  );
}
