import { Stack, Typography } from "@mui/material";

import { getVisualQueryParts, getVisualLabel } from "./LaypersonQueryUtils.js";

export default function QueryWizardReview({ value }) {
  const parts = getVisualQueryParts(value);
  return (
    <Stack spacing={1}>
      <Typography>
        <strong>Data:</strong> {parts.entity}
      </Typography>
      <Typography sx={{ overflowWrap: "anywhere" }}>
        <strong>Facets:</strong> {parts.dimensions}
      </Typography>
      <Typography>
        <strong>Visual:</strong> {getVisualLabel(parts.visual)}
      </Typography>
      <Typography>
        <strong>Calculate:</strong> {parts.aggregate}
      </Typography>
      <Typography variant="caption" sx={{ overflowWrap: "anywhere" }}>
        {value}
      </Typography>
    </Stack>
  );
}
