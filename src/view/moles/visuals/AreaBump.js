import { ResponsiveAreaBump } from "@nivo/bump";
import { Box, Typography } from "@mui/material";

import { NIVO_THEME } from "../../../AppTheme.js";
import { getMarkColor } from "../../../nonview/constants/COLORS.js";
import FormatUtils from "../visual_utils/FormatUtils.js";
import { toAreaBumpData } from "../visual_utils/AreaBumpData.js";

export {
  sortAreaBumpXAxis,
  toAreaBumpData,
} from "../visual_utils/AreaBumpData.js";

export default function AreaBump({ data, xAxisLabel }) {
  const series = toAreaBumpData(data);

  return (
    <Box sx={{ width: "100%", height: "100%", minHeight: 400 }}>
      <ResponsiveAreaBump
        theme={NIVO_THEME}
        data={series}
        margin={{ top: 40, right: 100, bottom: 60, left: 100 }}
        spacing={8}
        colors={(serie) => getMarkColor(serie.color)}
        blendMode="multiply"
        startLabel
        endLabel
        axisTop={null}
        axisBottom={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: -45,
          legend: xAxisLabel,
          legendPosition: "middle",
          legendOffset: 50,
        }}
        tooltip={({ serie }) => (
          <Typography variant="body2">
            {serie.id}:{" "}
            {FormatUtils.humanizeValue(
              serie.points.reduce((sum, point) => sum + point.data.y, 0),
            )}
          </Typography>
        )}
        role="img"
      />
    </Box>
  );
}

AreaBump.IS_CHART = true;
AreaBump.IS_STACKED = true;
AreaBump.IS_FULL_WIDTH = true;
