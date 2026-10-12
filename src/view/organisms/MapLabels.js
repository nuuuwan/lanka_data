import { useLayoutEffect, useRef, useState } from "react";

import {
  VISUAL_LABEL_MIN_FONT_SIZE,
  VISUAL_LABEL_MAX_FONT_SIZE,
} from "../../nonview/constants/ChartLabels.js";
import FormatUtils from "../moles/visual_utils/FormatUtils.js";

export default function MapLabels({ labels, testId }) {
  const ref = useRef(null);
  const [scale, setScale] = useState(0);
  useLayoutEffect(() => {
    const svg = ref.current.ownerSVGElement;
    const observer = new ResizeObserver(([entry]) => {
      setScale(entry.contentRect.width / svg.viewBox.baseVal.width);
    });
    observer.observe(svg);
    return () => observer.disconnect();
  }, []);

  return (
    <g ref={ref} data-testid={testId} pointerEvents="none">
      {scale > 0 &&
        labels.map((label) => {
          const { angle, fontSize, id, name } = label;
          const [x, y] = label.center ?? label.position;
          const size = Math.min(fontSize, VISUAL_LABEL_MAX_FONT_SIZE / scale);
          if (size * scale < VISUAL_LABEL_MIN_FONT_SIZE) return null;
          return (
            <text
              key={id}
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="central"
              fill={
                FormatUtils.isLightColor(label.color ?? label.backgroundColor)
                  ? "#000000"
                  : "#ffffff"
              }
              fontSize={size}
              transform={`rotate(${angle} ${x} ${y})`}
            >
              <title>{label.fullName ?? name}</title>
              {name}
            </text>
          );
        })}
    </g>
  );
}
