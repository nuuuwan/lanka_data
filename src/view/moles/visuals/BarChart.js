import { ResponsiveBar } from "@nivo/bar";
import { Box, Grid, Typography } from "@mui/material";

function getBarValue(datum) {
  const value = parseFloat(datum.answerThing.value);
  return Number.isNaN(value) ? 0 : value;
}

function getDimIndexInfo(datumList) {
  const nDims = datumList[0].query.dimThingList.length;
  const varyingDimIndexes = [];

  for (let dimIndex = 0; dimIndex < nDims; dimIndex++) {
    const values = new Set(
      datumList.map((datum) => datum.query.dimThingList[dimIndex].value),
    );
    if (values.size > 1) {
      varyingDimIndexes.push(dimIndex);
    }
  }

  return { nDims, varyingDimIndexes };
}

function getXAxisDimIndex(datumList) {
  const { varyingDimIndexes } = getDimIndexInfo(datumList);
  if (varyingDimIndexes.length === 0) {
    return 0;
  }
  return varyingDimIndexes[0];
}

function getFacetDimIndexes(datumList) {
  const xAxisDimIndex = getXAxisDimIndex(datumList);
  const { nDims, varyingDimIndexes } = getDimIndexInfo(datumList);
  return Array.from({ length: nDims }, (_, i) => i).filter(
    (dimIndex) =>
      dimIndex !== xAxisDimIndex && varyingDimIndexes.includes(dimIndex),
  );
}

function getFacetKey(datum, facetDimIndexes) {
  return facetDimIndexes
    .map((dimIndex) =>
      datum.query.dimThingList[dimIndex].getHumanReadableValue(),
    )
    .join(" / ");
}

function getXLabel(datum, xAxisDimIndex) {
  return datum.query.dimThingList[xAxisDimIndex].getHumanReadableValue();
}

function getBarColor(datum, xAxisDimIndex) {
  return datum.query.dimThingList[xAxisDimIndex].getColor();
}

function groupDataByFacet(datumList, xAxisDimIndex, facetDimIndexes) {
  const groups = new Map();

  for (const datum of datumList) {
    const facetKey = getFacetKey(datum, facetDimIndexes);
    if (!groups.has(facetKey)) {
      groups.set(facetKey, []);
    }
    groups.get(facetKey).push({
      id: getXLabel(datum, xAxisDimIndex),
      value: getBarValue(datum),
      color: getBarColor(datum, xAxisDimIndex),
    });
  }

  return Array.from(groups.entries()).map(([facetKey, data]) => ({
    facetKey,
    data,
  }));
}

function SingleBarChart({ data, xAxisLabel, yAxisLabel }) {
  return (
    <Box sx={{ height: 400 }}>
      <ResponsiveBar
        data={data}
        keys={["value"]}
        indexBy="id"
        margin={{ top: 50, right: 50, bottom: 100, left: 60 }}
        padding={0.3}
        valueScale={{ type: "linear" }}
        colors={(bar) => bar.data.color ?? "#1f77b4"}
        axisBottom={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: -45,
          legend: xAxisLabel,
          legendPosition: "middle",
          legendOffset: 80,
        }}
        axisLeft={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: yAxisLabel,
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

export default function BarChart({ datumSet }) {
  const { datumList } = datumSet;

  if (datumList.length === 0) {
    return <Typography>No data to display.</Typography>;
  }

  if (datumList.every((datum) => getBarValue(datum) === 0)) {
    return <Typography>Bar chart requires numeric values.</Typography>;
  }

  const xAxisDimIndex = getXAxisDimIndex(datumList);
  const facetDimIndexes = getFacetDimIndexes(datumList);
  const xAxisDimName =
    datumList[0].query.dimThingList[xAxisDimIndex].constructor.name;
  const yAxisLabel = datumList[0].query.aggregate;
  const facets = groupDataByFacet(datumList, xAxisDimIndex, facetDimIndexes);

  return (
    <Grid container spacing={2} sx={{ width: "100%" }}>
      {facets.map(({ facetKey, data }) => (
        <Grid item xs={12} md={facets.length > 1 ? 6 : 12} key={facetKey}>
          <Typography variant="h6" sx={{ mb: 1 }}>
            {facetKey || xAxisDimName}
          </Typography>
          <Box sx={{ width: "100%", minWidth: 0 }}>
            <SingleBarChart
              data={data}
              xAxisLabel={xAxisDimName}
              yAxisLabel={yAxisLabel}
            />
          </Box>
        </Grid>
      ))}
    </Grid>
  );
}
