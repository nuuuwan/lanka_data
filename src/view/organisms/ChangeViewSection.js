import { Box } from "@mui/material";

import VisualQueryForm from "./VisualQueryForm.js";

export default function ChangeViewSection({
  value,
  onChange,
  onSubmit,
  queryOptions,
}) {
  return (
    <Box sx={{ mt: 2 }}>
      <VisualQueryForm
        value={value}
        onChange={onChange}
        onSubmit={onSubmit}
        queryOptions={queryOptions}
      />
    </Box>
  );
}
