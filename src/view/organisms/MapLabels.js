import { useLayoutEffect, useRef, useState } from "react";

import {
  VISUAL_LABEL_MIN_FONT_SIZE,
  VISUAL_LABEL_MAX_FONT_SIZE,
} from "../../nonview/constants/ChartLabels.js";
import FormatUtils from "../moles/visual_utils/FormatUtils.js";
import StringUtils from "../../nonview/base/String.js";
import { getFittedLabelFontSize } from "../moles/visual_utils/GeoProjectionUtils.js";
import {
  MAP_LABEL_DARK_COLOR,
  MAP_LABEL_LIGHT_COLOR,
} from "../_cons/MapCons.js";

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
  }, [labels]);

  return (
    <g ref={ref} data-testid={testId} pointerEvents="none">
      {scale > 0 &&
        labels.map((label) => {
          const { angle, fontSize, id, name } = label;
          const [x, y] = label.center ?? label.position;
          let displayName = name;
          let fittedSize = fontSize;
          if (
            fontSize * scale < VISUAL_LABEL_MIN_FONT_SIZE &&
            label.width &&
            label.height
          ) {
            displayName = StringUtils.shorten(name, 3);
            fittedSize = getFittedLabelFontSize(
              displayName,
              label.width,
              label.height,
            );
          }
          const size = Math.min(fittedSize, VISUAL_LABEL_MAX_FONT_SIZE / scale);
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
                  ? MAP_LABEL_DARK_COLOR
                  : MAP_LABEL_LIGHT_COLOR
              }
              fontSize={size}
              transform={`rotate(${angle} ${x} ${y})`}
            >
              <title>{label.fullName ?? name}</title>
              {displayName}
            </text>
          );
        })}
    </g>
  );
}
