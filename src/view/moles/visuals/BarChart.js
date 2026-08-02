import { ResponsiveBar } from "@nivo/bar";
import { Box, Typography, useTheme } from "@mui/material";

import { FONT_FAMILY } from "../../../AppTheme.js";
import FormatUtils from "../visual_utils/FormatUtils.js";

export default function BarChart({ data, xAxisLabel, yAxisLabel }) {
  const theme = useTheme();

  return (
    <Box sx={{ height: 400 }}>
      <ResponsiveBar
        data={data}
        keys={["value"]}
        indexBy="id"
        margin={{ top: 50, right: 50, bottom: 100, left: 60 }}
        padding={0.3}
        valueScale={{ type: "linear" }}
        theme={{ fontFamily: FONT_FAMILY }}
        colors={(bar) => bar.data.color ?? theme.palette.primary.main}
        axisLeft={{
          format: FormatUtils.humanizeValue,
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: yAxisLabel,
          legendPosition: "middle",
          legendOffset: -50,
        }}
        label={({ value }) => FormatUtils.humanizeValue(value)}
        tooltip={({ value, indexValue }) => (
          <Typography variant="body2">
            {indexValue}: {FormatUtils.humanizeValue(value)}
          </Typography>
        )}
        axisBottom={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: -45,
          legend: xAxisLabel,
          legendPosition: "middle",
          legendOffset: 80,
        }}
        labelSkipWidth={12}
        labelSkipHeight={12}
        labelTextColor={{ from: "color", modifiers: [["darker", 1.6]] }}
        role="img"
        ariaLabel="Bar chart"
      />
    </Box>
  );
}

BarChart.IS_CHART = true;
