import { useState } from "react";
import {
  Box,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Typography,
} from "@mui/material";

export default function MultiChartLayout({
  facets,
  xAxisDimName,
  yAxisLabel,
  renderChart,
}) {
  const [selectedFacetKey, setSelectedFacetKey] = useState(
    facets[0]?.facetKey ?? "",
  );

  if (facets.length === 0) {
    return <Typography>No data to display.</Typography>;
  }

  const activeFacet =
    facets.find(({ facetKey }) => facetKey === selectedFacetKey) ?? facets[0];
  const activeFacetTitle = activeFacet.facetKey || xAxisDimName;

  return (
    <Box>
      {facets.length > 1 && (
        <FormControl size="small" sx={{ mb: 2, minWidth: 160 }}>
          <InputLabel id="facet-select-label">Facet</InputLabel>
          <Select
            label="Facet"
            labelId="facet-select-label"
            value={activeFacet.facetKey}
            onChange={({ target }) => setSelectedFacetKey(target.value)}
          >
            {facets.map(({ facetKey }) => (
              <MenuItem key={facetKey} value={facetKey}>
                {facetKey}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      )}
      <Typography component="h2" variant="h6" sx={{ mb: 1 }}>
        {activeFacetTitle}
      </Typography>
      <Box sx={{ width: "100%", minWidth: 0 }}>
        {renderChart({
          data: activeFacet.data,
          xAxisLabel: xAxisDimName,
          yAxisLabel,
        })}
      </Box>
    </Box>
  );
}
