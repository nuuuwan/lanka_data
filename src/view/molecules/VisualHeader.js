import { Box, Typography } from "@mui/material";

import getQueryFinding from "../../nonview/core/QueryFinding.js";
import FormatUtils from "../moles/visual_utils/FormatUtils.js";

export default function VisualHeader({
  query,
  encodedQuery,
  datumCount,
  loadTimeSeconds,
}) {
  return (
    <Box component="header" sx={{ mb: 2 }}>
      <Typography component="h1" variant="h4" data-testid="query-finding">
        {getQueryFinding(query)}
      </Typography>
      <Typography
        component="p"
        variant="body2"
        sx={{ color: "text.secondary", mt: 1, overflowWrap: "anywhere" }}
      >
        Query: <Box component="code">{encodedQuery}</Box>
      </Typography>
      <Typography
        data-testid="datums-count"
        component="p"
        variant="caption"
        sx={{ color: "text.secondary" }}
      >
        {datumCount} datums loaded in{" "}
        {FormatUtils.humanizeDuration(loadTimeSeconds)}
      </Typography>
    </Box>
  );
}
