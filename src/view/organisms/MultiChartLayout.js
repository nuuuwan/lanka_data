import { Box, Typography } from "@mui/material";

import Legend from "../moles/visuals/Legend.js";
import styles from "./MultiChartLayout.module.css";

const DIMENSION_LABELS = {
  DSD: "Divisional Secretariat Division",
  ED: "Electoral District",
  GND: "Grama Niladhari Division",
  PD: "Polling Division",
};

function humanizeTitle(value) {
  return value
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function humanizeDimension(value) {
  return DIMENSION_LABELS[value] ?? humanizeTitle(value);
}

function renderFacetTitle(facetKey) {
  return facetKey.split(" / ").flatMap((facet, index) => {
    const separator = index > 0 ? " / " : "";
    const delimiterIndex = facet.indexOf("=");
    if (delimiterIndex === -1) {
      return [separator, <strong key={facet}>{humanizeTitle(facet)}</strong>];
    }
    const dimension = facet.slice(0, delimiterIndex);
    const value = facet.slice(delimiterIndex + 1);
    return [
      separator,
      <strong key={facet}>{humanizeTitle(value)}</strong>,
      ` ${humanizeDimension(dimension)}`,
    ];
  });
}

export default function MultiChartLayout({
  facets,
  xAxisDimName,
  yAxisLabel,
  renderChart,
  fullWidth = false,
  legendItems,
}) {
  if (facets.length === 0) {
    return <Typography>No data to display.</Typography>;
  }

  return (
    <Box>
      <Box className={styles.grid}>
        {facets.map(({ facetKey, data, total }) => (
          <Box
            className={[
              styles.item,
              (fullWidth || facets.length === 1) && styles.fullWidth,
            ]
              .filter(Boolean)
              .join(" ")}
            key={facetKey}
          >
            <Typography
              component="h2"
              variant="h6"
              sx={{ mb: 1, textAlign: "center" }}
            >
              <Box component="span" sx={{ display: "block", width: "100%" }}>
                {facetKey
                  ? renderFacetTitle(facetKey)
                  : humanizeTitle(xAxisDimName)}
              </Box>
            </Typography>
            <Box sx={{ width: "100%", minWidth: 0 }}>
              {renderChart({
                data,
                total,
                xAxisLabel: xAxisDimName,
                yAxisLabel,
              })}
            </Box>
          </Box>
        ))}
      </Box>
      {legendItems?.length > 0 && <Legend items={legendItems} />}
    </Box>
  );
}
