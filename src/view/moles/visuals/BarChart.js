import { ResponsiveBar } from "@nivo/bar";
import { Box, Typography } from "@mui/material";

import { NIVO_THEME } from "../../../AppTheme.js";
import { getMarkColor } from "../../../nonview/constants/COLORS.js";
import FormatUtils from "../visual_utils/FormatUtils.js";
import InBarLabels from "./InBarLabels.js";

export default function BarChart({ data, xAxisLabel, yAxisLabel }) {
  return (
    <Box sx={{ width: "100%", height: "100%", minHeight: 400 }}>
      <ResponsiveBar
        theme={NIVO_THEME}
        data={data}
        keys={["value"]}
        indexBy="id"
        animate={false}
        margin={{ top: 50, right: 50, bottom: 100, left: 60 }}
        padding={0.3}
        valueScale={{ type: "linear" }}
        colors={(bar) => getMarkColor(bar.data.color)}
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
        enableLabel={false}
        layers={[
          "grid",
          "axes",
          "bars",
          InBarLabels,
          "totals",
          "markers",
          "legends",
          "annotations",
        ]}
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
        role="img"
        ariaLabel="Bar chart"
      />
    </Box>
  );
}

BarChart.IS_CHART = true;
