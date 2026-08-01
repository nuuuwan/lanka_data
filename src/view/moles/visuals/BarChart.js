import { ResponsiveBar } from "@nivo/bar";
import { Box, Typography } from "@mui/material";

function getBarLabel(datum) {
  return datum.query.dimThingList
    .map((thing) => thing.getHumanReadableValue())
    .join(" / ");
}

function getBarValue(datum) {
  const value = parseFloat(datum.answerThing.value);
  return Number.isNaN(value) ? 0 : value;
}

export default function BarChart({ datumSet }) {
  const { datumList } = datumSet;

  const data = datumList.map((datum) => ({
    id: getBarLabel(datum) || datum.query.aggregate,
    value: getBarValue(datum),
  }));

  if (data.length === 0) {
    return <Typography>No data to display.</Typography>;
  }

  if (data.every((item) => item.value === 0)) {
    return <Typography>Bar chart requires numeric values.</Typography>;
  }

  return (
    <Box sx={{ height: 400 }}>
      <ResponsiveBar
        data={data}
        keys={["value"]}
        indexBy="id"
        margin={{ top: 50, right: 50, bottom: 100, left: 60 }}
        padding={0.3}
        valueScale={{ type: "linear" }}
        colors={{ scheme: "nivo" }}
        axisBottom={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: -45,
          legend: "Dimension",
          legendPosition: "middle",
          legendOffset: 80,
        }}
        axisLeft={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: "Value",
          legendPosition: "middle",
          legendOffset: -50,
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
