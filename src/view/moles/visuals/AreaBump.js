import { ResponsiveAreaBump } from "@nivo/bump";
import { Box, Typography, useTheme } from "@mui/material";

import { FONT_FAMILY } from "../../../AppTheme.js";
import FormatUtils from "../visual_utils/FormatUtils.js";

function getSeriesKeys(data) {
  if (data.length === 0) {
    return [];
  }
  return Object.keys(data[0]).filter(
    (key) => key !== "id" && key !== "_barWidth" && !key.endsWith("Color"),
  );
}

export function toAreaBumpData(data) {
  return getSeriesKeys(data).map((key) => ({
    id: key,
    color: data.find((row) => row[`${key}Color`])?.[`${key}Color`],
    data: data.map((row) => ({
      x: row.id,
      y: row[key] ?? 0,
    })),
  }));
}

export default function AreaBump({ data, xAxisLabel }) {
  const theme = useTheme();
  const series = toAreaBumpData(data);

  return (
    <Box sx={{ height: 400 }}>
      <ResponsiveAreaBump
        data={series}
        margin={{ top: 40, right: 100, bottom: 60, left: 100 }}
        spacing={8}
        colors={(serie) => serie.color ?? theme.palette.primary.main}
        blendMode="multiply"
        theme={{ fontFamily: FONT_FAMILY }}
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
