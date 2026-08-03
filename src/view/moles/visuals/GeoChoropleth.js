import { Box } from "@mui/material";
import { Choropleth } from "@nivo/geo";

import { NIVO_THEME } from "../../../AppTheme.js";
import {
  MAP_BORDER_COLOR,
  MAP_BORDER_WIDTH,
  MAP_HEIGHT,
  MAP_LABEL_DARK_COLOR,
  MAP_LABEL_LIGHT_COLOR,
  MAP_UNKNOWN_COLOR,
  MAP_WIDTH,
} from "../../_cons/MapCons.js";
import FormatUtils from "../visual_utils/FormatUtils.js";

function setViewBox(element) {
  element
    ?.querySelector("svg")
    ?.setAttribute("viewBox", `0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`);
}

export default function GeoChoropleth({
  testId,
  features,
  data,
  labels,
  projectionScale,
  projectionTranslation,
}) {
  const maxValue = Math.max(...data.map(({ value }) => value), 1);
  const labelsLayer = () => (
    <g data-testid={`${testId}-labels`} pointerEvents="none">
      {labels.map(
        ({ angle, backgroundColor, fontSize, id, name, position: [x, y] }) => (
          <text
            key={id}
            x={x}
            y={y}
            textAnchor="middle"
            dominantBaseline="central"
            fill={
              FormatUtils.isLightColor(backgroundColor)
                ? MAP_LABEL_DARK_COLOR
                : MAP_LABEL_LIGHT_COLOR
            }
            fontSize={fontSize}
            transform={`rotate(${angle} ${x} ${y})`}
          >
            {name}
          </text>
        ),
      )}
    </g>
  );

  return (
    <Box
      ref={setViewBox}
      data-testid={testId}
      sx={{
        width: "100%",
        "& svg": { width: "100%", height: "auto", display: "block" },
      }}
    >
      <Choropleth
        theme={NIVO_THEME}
        width={MAP_WIDTH}
        height={MAP_HEIGHT}
        features={features}
        data={data}
        domain={[0, maxValue]}
        label={(mapFeature) =>
          mapFeature.data
            ? `${mapFeature.properties.name}: ${mapFeature.data.categoryLabel}`
            : mapFeature.properties.name
        }
        valueFormat={FormatUtils.humanizeValue}
        projectionType="mercator"
        projectionScale={projectionScale}
        projectionTranslation={projectionTranslation}
        colors={[MAP_UNKNOWN_COLOR, MAP_UNKNOWN_COLOR]}
        unknownColor={MAP_UNKNOWN_COLOR}
        borderWidth={MAP_BORDER_WIDTH}
        borderColor={MAP_BORDER_COLOR}
        layers={labels.length > 0 ? ["features", labelsLayer] : ["features"]}
        role="img"
      />
    </Box>
  );
}
