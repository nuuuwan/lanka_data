import { Box } from "@mui/material";

import {
  MAP_BORDER_COLOR,
  MAP_WIDTH,
  MAP_UNKNOWN_COLOR,
  SHAPE_MAP_EDGE_WIDTH,
  SHAPE_MAP_REGION_BORDER_WIDTH,
} from "../../_cons/MapCons.js";
import FormatUtils from "../visual_utils/FormatUtils.js";
import MapLabels from "../../organisms/MapLabels.js";

export default function ShapeMapGraphic({ data, shapeConfig }) {
  const fillColor = data.facetColor ?? MAP_UNKNOWN_COLOR;
  return (
    <Box
      data-testid={shapeConfig.testId}
      sx={{
        width: "100%",
        maxWidth: MAP_WIDTH,
        mx: "auto",
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
            fill={shape.display.color ?? fillColor}
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
          <MapLabels labels={data.labels} />
        </g>
      </svg>
    </Box>
  );
}
