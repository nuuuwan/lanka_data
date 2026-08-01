import { ResponsiveMarimekko } from "@nivo/marimekko";
import { Box, Typography } from "@mui/material";

function getColorForDimension(id, data) {
  const colorKey = `${id}Color`;
  for (const row of data) {
    if (row[colorKey]) {
      return row[colorKey];
    }
  }
  return undefined;
}

export default function MarimekkoChart({
  data,
  xAxisLabel,
  yAxisLabel,
  stackDimName,
}) {
  const dimensions =
    data.length === 0
      ? []
      : Object.keys(data[0])
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
        data={data}
        id="id"
        value="_barWidth"
        dimensions={dimensions}
        offset="expand"
        layout="vertical"
        axisTop={null}
        axisRight={null}
        enableGridY={false}
        axisBottom={{
          orient: "bottom",
          tickSize: 5,
          tickPadding: 5,
          tickRotation: -45,
          legend: xAxisLabel,
          legendOffset: 80,
          legendPosition: "middle",
          format: (value) => value,
        }}
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
        colors={({ id }) => getColorForDimension(id, data) ?? "#1f77b4"}
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
