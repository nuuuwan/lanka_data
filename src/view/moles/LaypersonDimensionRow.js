import DeleteOutlinedIcon from "@mui/icons-material/DeleteOutlined";
import {
  Box,
  IconButton,
  ListSubheader,
  MenuItem,
  TextField,
} from "@mui/material";

import {
  DIMENSION_OPERATORS,
  getFieldGroups,
} from "../../nonview/constants/VisualQueryOptions.js";
import LaypersonDimensionValue from "./LaypersonDimensionValue.js";
import { getVisualLabel } from "./LaypersonQueryUtils.js";

export default function LaypersonDimensionRow({
  dimension,
  dimensionOptions,
  dimensionsCount,
  index,
  onChange,
  onKeyDown,
  onRemove,
}) {
  const fieldOptions = dimensionOptions.includes(dimension.field)
    ? dimensionOptions
    : [dimension.field, ...dimensionOptions].filter(Boolean);

  return (
    <Box
      sx={{
        alignItems: "center",
        display: "grid",
        gap: 0.75,
        gridTemplateColumns: {
          xs: "minmax(0, 1fr) auto",
          sm: "minmax(10rem, 2fr) minmax(7rem, 1fr) minmax(10rem, 2fr) auto",
        },
        mb: 0.75,
      }}
    >
      <TextField
        select
        label={index === 0 ? "Field" : undefined}
        size="small"
        slotProps={{ htmlInput: { "aria-label": "Field" } }}
        value={dimension.field}
        onChange={(event) => onChange("field", event.target.value)}
      >
        {getFieldGroups(fieldOptions).flatMap((group) => [
          <ListSubheader key={`group-${group.label}`}>
            {group.label}
          </ListSubheader>,
          ...group.fields.map((field) => (
            <MenuItem key={field} value={field}>
              {getVisualLabel(field)}
            </MenuItem>
          )),
        ])}
      </TextField>
      <TextField
        select
        label={index === 0 ? "Operator" : undefined}
        size="small"
        slotProps={{
          htmlInput: { "aria-label": "Operator" },
          inputLabel: { shrink: true },
          select: { displayEmpty: true },
        }}
        value={dimension.operator}
        onChange={(event) => onChange("operator", event.target.value)}
      >
        {DIMENSION_OPERATORS.map((operator) => (
          <MenuItem key={operator.label} value={operator.value}>
            {operator.label}
          </MenuItem>
        ))}
      </TextField>
      <LaypersonDimensionValue
        dimension={dimension}
        dimensionOptions={dimensionOptions}
        index={index}
        onChange={onChange}
        onKeyDown={onKeyDown}
      />
      <IconButton
        aria-label={`Remove condition ${index + 1}`}
        disabled={dimensionsCount === 1}
        onClick={onRemove}
        size="small"
        sx={{
          color: "error.dark",
          "&:hover": {
            bgcolor: "error.light",
          },
        }}
      >
        <DeleteOutlinedIcon />
      </IconButton>
    </Box>
  );
}
