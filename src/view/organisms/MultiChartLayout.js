import { useState } from "react";
import {
  Box,
  Checkbox,
  FormControl,
  InputLabel,
  ListItemText,
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
  const [selectedFacetKeys, setSelectedFacetKeys] = useState(() =>
    facets.map(({ facetKey }) => facetKey),
  );

  if (facets.length === 0) {
    return <Typography>No data to display.</Typography>;
  }

  const selectedFacetKeySet = new Set(selectedFacetKeys);
  const selectedFacets = facets.filter(({ facetKey }) =>
    selectedFacetKeySet.has(facetKey),
  );

  return (
    <Box>
      {facets.length > 1 && (
        <FormControl size="small" sx={{ mb: 2, minWidth: 160 }}>
          <InputLabel id="facet-select-label">Facets</InputLabel>
          <Select
            label="Facets"
            labelId="facet-select-label"
            multiple
            value={selectedFacetKeys}
            onChange={({ target }) => setSelectedFacetKeys(target.value)}
            renderValue={(selected) =>
              selected.length === 0 ? "None" : selected.join(", ")
            }
          >
            {facets.map(({ facetKey }) => (
              <MenuItem key={facetKey} value={facetKey}>
                <Checkbox checked={selectedFacetKeySet.has(facetKey)} />
                <ListItemText primary={facetKey} />
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      )}
      {selectedFacets.map(({ facetKey, data }) => (
        <Box key={facetKey}>
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
