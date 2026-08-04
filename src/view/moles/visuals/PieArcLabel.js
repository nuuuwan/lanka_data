import { animated } from "@react-spring/web";

import FormatUtils from "../visual_utils/FormatUtils.js";

const CHARACTER_WIDTH_RATIO = 0.6;
const LABEL_PADDING_RATIO = 0.6;
const MAX_FONT_SIZE = 28;
const MIN_FONT_SIZE = 10;

function getFontSize(
  { endAngle, innerRadius, outerRadius, startAngle },
  label,
) {
  const middleRadius = (innerRadius + outerRadius) / 2;
  const arcLength = middleRadius * Math.abs(endAngle - startAngle);
  const radialSpace = outerRadius - innerRadius;
  const textWidth = Math.max(label.length, 1) * CHARACTER_WIDTH_RATIO;
  const fit = Math.min(radialSpace, arcLength / textWidth);
  return Math.min(
    MAX_FONT_SIZE,
    Math.max(MIN_FONT_SIZE, fit * LABEL_PADDING_RATIO),
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
