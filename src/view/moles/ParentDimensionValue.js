import { Box, ListSubheader, MenuItem, TextField } from "@mui/material";

import {
  DIMENSION_OPERATORS,
  getFieldGroups,
} from "../../nonview/constants/VisualQueryOptions.js";
import LaypersonDimensionValue from "./LaypersonDimensionValue.js";
import {
  getDimensionParts,
  getDimensionString,
  getVisualLabel,
} from "./LaypersonQueryUtils.js";

export default function ParentDimensionValue({
  dimension,
  dimensionOptions,
  onChange,
  onKeyDown,
}) {
  const [parent] = getDimensionParts(dimension.value);
  const fields = dimensionOptions.includes(parent.field)
    ? dimensionOptions
    : [parent.field, ...dimensionOptions].filter(Boolean);
  function update(name, value) {
    const next = { ...parent, [name]: value };
    if (name === "field" && !next.operator) next.operator = "=";
    onChange("value", getDimensionString(next));
  }
  return (
    <Box
      sx={{
        display: "grid",
        gap: 0.75,
        gridTemplateColumns: {
          xs: "1fr",
          sm: "minmax(8rem, 1fr) minmax(5rem, auto) minmax(8rem, 1fr)",
        },
      }}
    >
      <TextField
        select
        label="Parent field"
        size="small"
        slotProps={{ htmlInput: { "aria-label": "Parent field" } }}
        value={parent.field}
        onChange={(event) => update("field", event.target.value)}
      >
        {getFieldGroups(fields).flatMap((group) => [
          <ListSubheader key={`group-${group.label}`}>
            {group.label}
          </ListSubheader>,
          ...group.fields.map((field) => (
            <MenuItem key={field} sx={{ pl: 3 }} value={field}>
              {getVisualLabel(field)}
            </MenuItem>
          )),
        ])}
      </TextField>
      <TextField
        select
        label="Operator"
        size="small"
        slotProps={{ htmlInput: { "aria-label": "Parent operator" } }}
        value={parent.operator}
        onChange={(event) => update("operator", event.target.value)}
      >
        {DIMENSION_OPERATORS.filter(
          ({ value }) => value !== "<" && value !== "",
        ).map((operator) => (
          <MenuItem key={operator.label} value={operator.value}>
            {operator.label}
          </MenuItem>
        ))}
      </TextField>
      <LaypersonDimensionValue
        dimension={parent}
        dimensionOptions={dimensionOptions}
        index={0}
        onChange={update}
        onKeyDown={onKeyDown}
      />
    </Box>
  );
}
