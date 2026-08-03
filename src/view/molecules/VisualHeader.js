import { Box, Typography } from "@mui/material";

import { getQueryFindingParts } from "../../nonview/core/QueryFinding.js";
import FormatUtils from "../moles/visual_utils/FormatUtils.js";

export default function VisualHeader({
  query,
  datumCount,
  loadTimeSeconds,
  titleRef,
}) {
  const titleParts = getQueryFindingParts(query);

  return (
    <Box component="header" ref={titleRef} sx={{ mb: 2 }}>
      <Typography component="h1" variant="h4" data-testid="query-finding">
        {titleParts.map((part, index) =>
          typeof part === "string" ? (
            part
          ) : (
            <strong key={index}>{part.text}</strong>
          ),
        )}
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
