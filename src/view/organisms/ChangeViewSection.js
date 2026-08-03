import { Box } from "@mui/material";

import VisualQueryForm from "./VisualQueryForm.js";

export default function ChangeViewSection({
  disabled,
  value,
  onChange,
  onSubmit,
  queryOptions,
}) {
  return (
    <Box sx={{ mb: 4 }}>
      <VisualQueryForm
        disabled={disabled}
        value={value}
        onChange={onChange}
        onSubmit={onSubmit}
        queryOptions={queryOptions}
      />
    </Box>
  );
}
