import AddIcon from "@mui/icons-material/Add";
import { Box, Button, Typography } from "@mui/material";

import LaypersonDimensionRow from "./LaypersonDimensionRow.js";

export default function LaypersonDimensions({
  dimensions,
  dimensionOptions,
  onAdd,
  onChange,
  onKeyDown,
  onRemove,
}) {
  return (
    <Box
      sx={{
        bgcolor: "background.paper",
        border: 1,
        borderColor: "divider",
        borderRadius: 1,
        boxShadow: 1,
        gridColumn: "1 / -1",
        p: 1.5,
      }}
    >
      <Typography component="div" variant="subtitle2" sx={{ mb: 1 }}>
        Group or filter
      </Typography>
      {dimensions.map((dimension, index) => (
        <LaypersonDimensionRow
          key={index}
          dimension={dimension}
          dimensionOptions={dimensionOptions}
          dimensionsCount={dimensions.length}
          index={index}
          onChange={(name, value) => onChange(index, name, value)}
          onKeyDown={onKeyDown}
          onRemove={() => onRemove(index)}
        />
      ))}
      <Button
        disabled={!dimensions.at(-1).field}
        onClick={onAdd}
        size="small"
        startIcon={<AddIcon />}
      >
        AND
      </Button>
    </Box>
  );
}
