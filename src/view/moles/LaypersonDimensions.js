import AddIcon from "@mui/icons-material/Add";
import { Button, Paper, Typography } from "@mui/material";

import LaypersonDimensionRow from "./LaypersonDimensionRow.js";

export default function LaypersonDimensions({
  dimensions,
  dimensionOptions,
  valueOptionsByField,
  onAdd,
  onChange,
  onKeyDown,
  onRemove,
}) {
  return (
    <Paper
      elevation={0}
      sx={{
        bgcolor: "grey.100",
        gridColumn: "1 / -1",
        p: 1.5,
      }}
    >
      <Typography component="div" variant="subtitle2" sx={{ mb: 1 }}>
        Facets
      </Typography>
      {dimensions.map((dimension, index) => (
        <LaypersonDimensionRow
          key={index}
          dimension={dimension}
          dimensionOptions={dimensionOptions}
          valueOptionsByField={valueOptionsByField}
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
    </Paper>
  );
}
