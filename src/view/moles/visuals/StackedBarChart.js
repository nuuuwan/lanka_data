import { ResponsiveBar } from "@nivo/bar";
import { Box, Typography } from "@mui/material";

import { NIVO_THEME } from "../../../AppTheme.js";
import { getMarkColor } from "../../../nonview/constants/COLORS.js";
import {
  BAR_CHART_MIN_HEIGHT_PX,
  getDenseAxisTheme,
  getSlantedXAxis,
  X_AXIS_BOTTOM_MARGIN,
} from "../visual_utils/ChartAxisUtils.js";
import FormatUtils from "../visual_utils/FormatUtils.js";
import InBarLabels from "./InBarLabels.js";
import Legend from "./Legend.js";

function getColorForKey(key, data) {
  const colorKey = `${key}Color`;
  for (const row of data) {
    if (row[colorKey]) {
      return row[colorKey];
    }
  }
  return undefined;
}

export default function StackedBarChart({
  data,
  xAxisLabel,
  yAxisLabel,
  stackDimName,
}) {
  const keys =
    data.length === 0
      ? []
      : Object.keys(data[0]).filter(
          (key) =>
            key !== "id" && key !== "_barWidth" && !key.endsWith("Color"),
        );

  return (
    <Box>
      <Box
        sx={{
          width: "100%",
          height: "100%",
          minHeight: BAR_CHART_MIN_HEIGHT_PX,
        }}
      >
        <ResponsiveBar
          theme={getDenseAxisTheme(NIVO_THEME, data.length)}
          data={data}
          keys={keys}
          indexBy="id"
          animate={false}
          margin={{
            top: 50,
            right: 50,
            bottom: X_AXIS_BOTTOM_MARGIN,
            left: 70,
          }}
          padding={0.3}
          valueScale={{ type: "linear" }}
          colors={({ id }) => getMarkColor(getColorForKey(id, data))}
          borderColor={{ from: "color", modifiers: [["darker", 1.6]] }}
          axisLeft={{
            format: FormatUtils.humanizeValue,
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: yAxisLabel,
            legendPosition: "middle",
            legendOffset: -50,
          }}
          axisBottom={getSlantedXAxis(xAxisLabel)}
          tooltip={({ value, indexValue, id }) => (
            <Typography variant="body2">
              {indexValue} - {id}: {FormatUtils.humanizeValue(value)}
            </Typography>
          )}
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
          role="img"
          ariaLabel="Stacked bar chart"
        />
      </Box>
      <Legend
        items={keys.map((key) => ({
          id: key,
          label: key,
          color: getMarkColor(getColorForKey(key, data)),
        }))}
      />
    </Box>
  );
}

StackedBarChart.IS_CHART = true;
StackedBarChart.IS_FULL_WIDTH = true;
StackedBarChart.IS_STACKED = true;
