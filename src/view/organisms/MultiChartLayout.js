import { Box, Typography } from "@mui/material";

import styles from "./MultiChartLayout.module.css";

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
    <Box className={styles.grid}>
      {facets.map(({ facetKey, data }) => (
        <Box className={styles.item} key={facetKey}>
          <Typography component="h2" variant="h6" sx={{ mb: 1 }}>
            {facetKey || xAxisDimName}
          </Typography>
          <Box sx={{ width: "100%", minWidth: 0 }}>
            {renderChart({
              data,
              xAxisLabel: xAxisDimName,
              yAxisLabel,
            })}
          </Box>
        </Box>
      ))}
    </Box>
  );
}
