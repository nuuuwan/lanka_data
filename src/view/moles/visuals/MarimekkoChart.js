import { ResponsiveMarimekko } from "@nivo/marimekko";
import { Box, Typography, useTheme } from "@mui/material";

import { FONT_FAMILY } from "../../../AppTheme.js";

function getColorForDimension(id, data) {
  const colorKey = `${id}Color`;
  for (const row of data) {
    if (row[colorKey]) {
      return row[colorKey];
    }
  }
  return undefined;
}

function getDimensionKeys(row) {
  return Object.keys(row).filter(
    (key) => key !== "id" && key !== "_barWidth" && !key.endsWith("Color"),
  );
}

function getDominantKey(data) {
  if (data.length === 0) return null;
  const keys = getDimensionKeys(data[0]);
  const totals = Object.fromEntries(keys.map((key) => [key, 0]));
  for (const row of data) {
    for (const key of keys) {
      totals[key] += row[key] || 0;
    }
  }
  return keys.reduce((best, key) => (totals[key] > totals[best] ? key : best));
}

function getPercentage(row, key) {
  const keys = getDimensionKeys(row);
  const total = keys.reduce((sum, k) => sum + (row[k] || 0), 0);
  if (total === 0) return 0;
  return (row[key] || 0) / total;
}

function sortByDominantCategory(data) {
  const dominantKey = getDominantKey(data);
  if (!dominantKey) return data;
  return [...data].sort(
    (a, b) => getPercentage(b, dominantKey) - getPercentage(a, dominantKey),
  );
}

function BarLabelsLayer({ data }) {
  return (
    <>
      {data.map((datum) => (
        <text
          key={datum.id}
          x={datum.x + datum.width / 2}
          y={datum.y + datum.height + 16}
          textAnchor="middle"
          dominantBaseline="hanging"
          style={{
            fontFamily: FONT_FAMILY,
            fontSize: 12,
            fill: "#333",
            fontWeight: 500,
          }}
        >
          {datum.id}
        </text>
      ))}
    </>
  );
}

export default function MarimekkoChart({
  data,
  xAxisLabel,
  yAxisLabel,
  stackDimName,
}) {
  const theme = useTheme();
  const sortedData = sortByDominantCategory(data);

  const dimensions =
    sortedData.length === 0
      ? []
      : Object.keys(sortedData[0])
          .filter(
            (key) =>
              key !== "id" && key !== "_barWidth" && !key.endsWith("Color"),
          )
          .map((key) => ({
            id: key,
            value: key,
          }));

  return (
    <Box sx={{ height: 400 }}>
      <ResponsiveMarimekko
        data={sortedData}
        id="id"
        value="_barWidth"
        dimensions={dimensions}
        offset="expand"
        layout="vertical"
        theme={{
          fontFamily: FONT_FAMILY,
          text: { fontFamily: FONT_FAMILY },
        }}
        axisTop={null}
        axisRight={null}
        enableGridY={false}
        axisBottom={{
          orient: "bottom",
          tickSize: 0,
          tickPadding: 0,
          tickRotation: 0,
          legend: xAxisLabel,
          legendOffset: 40,
          legendPosition: "middle",
          format: () => "",
        }}
        layers={["grid", "axes", "bars", BarLabelsLayer, "legends"]}
        axisLeft={{
          orient: "left",
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: "%",
          legendOffset: -50,
          legendPosition: "middle",
          format: (value) => `${Math.round(value * 100)}%`,
        }}
        colors={({ id }) =>
          getColorForDimension(id, sortedData) ?? theme.palette.primary.main
        }
        borderWidth={1}
        borderColor={{ from: "color", modifiers: [["darker", 0.2]] }}
        margin={{ top: 40, right: 130, bottom: 100, left: 80 }}
        legends={[
          {
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
        tooltip={({ id, value, color }) => (
          <Box sx={{ backgroundColor: "white", p: 1, border: 1 }}>
            <Typography variant="body2" sx={{ color }}>
              {id}: {Math.round(value * 100)}%
            </Typography>
          </Box>
        )}
      />
    </Box>
  );
}

MarimekkoChart.IS_CHART = true;
MarimekkoChart.IS_STACKED = true;
MarimekkoChart.IS_MARIMEKKO = true;
