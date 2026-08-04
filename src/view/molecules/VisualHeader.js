import { Box, Typography } from "@mui/material";

import { VISUAL_CONTENT_MAX_WIDTH_PX } from "../../nonview/constants/APP.js";
import getQueryFinding from "../../nonview/core/QueryFinding.js";
import FormatUtils from "../moles/visual_utils/FormatUtils.js";

export default function VisualHeader({
  query,
  datumCount,
  loadTimeSeconds,
  titleRef,
}) {
  return (
    <Box
      component="header"
      ref={titleRef}
      sx={{
        mb: 2,
        maxWidth: VISUAL_CONTENT_MAX_WIDTH_PX,
        mx: "auto",
        textAlign: "center",
      }}
    >
      <Typography component="h1" variant="h4" data-testid="query-finding">
        {getQueryFinding(query)}
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
