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
    <Box
      sx={{
        display: "flex",
        flexWrap: "wrap",
        gap: 2,
      }}
    >
      {facets.map(({ facetKey, data }) => (
        <Box
          key={facetKey}
          sx={{
            width: {
              xs: "100%",
              sm: "calc(50% - 8px)",
              md: "calc(33.333% - 11px)",
            },
          }}
        >
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
