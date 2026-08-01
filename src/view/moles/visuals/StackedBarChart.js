import { ResponsiveBar } from "@nivo/bar";
import { Box, Typography } from "@mui/material";

import FormatUtils from "../visual_utils/FormatUtils.js";

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
          (key) => key !== "id" && !key.endsWith("Color"),
        );

  return (
    <Box sx={{ height: 400 }}>
      <ResponsiveBar
        data={data}
        keys={keys}
        indexBy="id"
        margin={{ top: 50, right: 130, bottom: 100, left: 60 }}
        padding={0.3}
        valueScale={{ type: "linear" }}
        colors={({ id }) => getColorForKey(id, data) ?? "#1f77b4"}
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
        axisBottom={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: -45,
          legend: xAxisLabel,
          legendPosition: "middle",
          legendOffset: 80,
        }}
        legends={[
          {
            dataFrom: "keys",
            anchor: "bottom-right",
            direction: "column",
            justify: false,
            translateX: 120,
            translateY: 0,
            itemsSpacing: 2,
            itemWidth: 100,
            itemHeight: 20,
            itemDirection: "left-to-right",
            itemOpacity: 0.85,
            symbolSize: 20,
          },
        ]}
        tooltip={({ value, indexValue, id }) => (
          <Typography variant="body2">
            {indexValue} - {id}: {FormatUtils.humanizeValue(value)}
          </Typography>
        )}
        labelSkipWidth={12}
        labelSkipHeight={12}
        labelTextColor={{ from: "color", modifiers: [["darker", 1.6]] }}
        role="img"
        ariaLabel="Stacked bar chart"
      />
    </Box>
  );
}

StackedBarChart.IS_CHART = true;
StackedBarChart.IS_STACKED = true;
