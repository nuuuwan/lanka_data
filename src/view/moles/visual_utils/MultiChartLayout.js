import { Box, Typography, Grid } from "@mui/material";

export default function MultiChartLayout({
  facets,
  xAxisDimName,
  yAxisLabel,
  renderChart,
}) {
  if (facets.length === 0) {
    return <Typography>No data to display.</Typography>;
  }

  return (
    <Grid container spacing={1}>
      {facets.map(({ facetKey, data }) => (
        <Grid key={facetKey} size={{ xs: 12, sm: 6, md: 4 }}>
          <Typography variant="title" sx={{ mb: 1 }}>
            {facetKey || xAxisDimName}
          </Typography>
          <Box sx={{ width: "100%", minWidth: 0 }}>
            {renderChart({ data, xAxisLabel: xAxisDimName, yAxisLabel })}
          </Box>
        </Grid>
      ))}
    </Grid>
  );
}
