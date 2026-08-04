import { Box } from "@mui/material";

import {
  MAP_BORDER_COLOR,
  MAP_LABEL_DARK_COLOR,
  MAP_LABEL_LIGHT_COLOR,
  MAP_UNKNOWN_COLOR,
  SHAPE_MAP_EDGE_WIDTH,
  SHAPE_MAP_REGION_BORDER_WIDTH,
} from "../../../nonview/constants/MAP.js";
import FormatUtils from "../../../nonview/core/visual/FormatUtils.js";

export default function ShapeMapGraphic({ data, shapeConfig }) {
  return (
    <Box
      data-testid={shapeConfig.testId}
      sx={{
        width: "100%",
        "& svg": { width: "100%", height: "auto", display: "block" },
      }}
    >
      <svg
        viewBox={data.viewBox.join(" ")}
        role="img"
        aria-label={shapeConfig.ariaLabel}
      >
        {data.shapes.map((shape) => (
          <polygon
            key={shape.id}
            points={shape.points.map(([x, y]) => `${x},${y}`).join(" ")}
            fill={shape.display.color ?? MAP_UNKNOWN_COLOR}
            stroke={MAP_BORDER_COLOR}
            strokeWidth={SHAPE_MAP_EDGE_WIDTH}
          >
            <title>
              {shape.feature.properties.name}: {shape.display.label} (
              {FormatUtils.humanizeValue(shape.display.value)})
            </title>
          </polygon>
        ))}
        <g pointerEvents="none">
          {data.boundaryEdges.map(({ start, end }, index) => (
            <line
              key={`${start.join(",")}-${end.join(",")}-${index}`}
              x1={start[0]}
              y1={start[1]}
              x2={end[0]}
              y2={end[1]}
              stroke={MAP_BORDER_COLOR}
              strokeWidth={SHAPE_MAP_REGION_BORDER_WIDTH}
            />
          ))}
          {data.labels.map(({ angle, center, color, fontSize, id, name }) => (
            <text
              key={id}
              x={center[0]}
              y={center[1]}
              textAnchor="middle"
              dominantBaseline="central"
              fill={
                FormatUtils.isLightColor(color)
                  ? MAP_LABEL_DARK_COLOR
                  : MAP_LABEL_LIGHT_COLOR
              }
              fontSize={fontSize}
              transform={`rotate(${angle} ${center[0]} ${center[1]})`}
            >
              {name}
            </text>
          ))}
        </g>
      </svg>
    </Box>
  );
}
