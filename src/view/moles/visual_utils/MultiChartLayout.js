import { Box, Typography } from "@mui/material";

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
    <Box>
      {facets.map(({ facetKey, data }) => (
        <Box key={facetKey}>
          <Typography variant="title" sx={{ mb: 1 }}>
            {facetKey || xAxisDimName}
          </Typography>
          <Box sx={{ width: "100%" }}>
            {renderChart({ data, xAxisLabel: xAxisDimName, yAxisLabel })}
          </Box>
        </Box>
      ))}
    </Box>
  );
}
