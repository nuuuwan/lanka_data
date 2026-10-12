import AddIcon from "@mui/icons-material/Add";
import { Box, IconButton, Paper, Tooltip } from "@mui/material";

import LaypersonDimensionRow from "./LaypersonDimensionRow.js";
import styles from "./LaypersonDimensions.module.css";

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
        bgcolor: "grey.200",
        gridColumn: "1 / -1",
        p: 1.5,
      }}
    >
      {dimensions.map((dimension, index) => (
        <Box key={index} className={styles.row}>
          <LaypersonDimensionRow
            dimension={dimension}
            dimensionOptions={dimensionOptions}
            valueOptionsByField={valueOptionsByField}
            dimensionsCount={dimensions.length}
            index={index}
            onChange={(name, value) => onChange(index, name, value)}
            onKeyDown={onKeyDown}
            onRemove={() => onRemove(index)}
          />
        </Box>
      ))}
      <Tooltip title="Add facet (AND)">
        <span>
          <IconButton
            aria-label="Add facet"
            disabled={!dimensions.at(-1).field}
            onClick={onAdd}
            size="small"
          >
            <AddIcon />
          </IconButton>
        </span>
      </Tooltip>
    </Paper>
  );
}
