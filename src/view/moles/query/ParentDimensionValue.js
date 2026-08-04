import { Box, ListSubheader, MenuItem, TextField } from "@mui/material";

import {
  DIMENSION_OPERATORS,
  getFieldGroups,
} from "../../../nonview/constants/VisualQueryOptions.js";
import { PARENT_DIMENSION_SX } from "../../../nonview/constants/VisualQueryLayout.js";
import {
  getDimensionParts,
  getDimensionString,
  getVisualLabel,
} from "./LaypersonQueryUtils.js";

export default function ParentDimensionValue({
  ValueComponent,
  dimension,
  dimensionOptions,
  onChange,
  onKeyDown,
}) {
  const [parentDimension] = getDimensionParts(dimension.value);
  const parentFieldOptions = dimensionOptions.includes(parentDimension.field)
    ? dimensionOptions
    : [parentDimension.field, ...dimensionOptions].filter(Boolean);
  const updateParentDimension = (name, value) => {
    const nextParentDimension = { ...parentDimension, [name]: value };
    if (name === "field" && !nextParentDimension.operator) {
      nextParentDimension.operator = "=";
    }
    onChange("value", getDimensionString(nextParentDimension));
  };

  return (
    <Box sx={PARENT_DIMENSION_SX}>
      <TextField
        select
        label="Parent field"
        size="small"
        slotProps={{ htmlInput: { "aria-label": "Parent field" } }}
        value={parentDimension.field}
        onChange={(event) => updateParentDimension("field", event.target.value)}
      >
        {getFieldGroups(parentFieldOptions).flatMap((group) => [
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
        value={parentDimension.operator}
        onChange={(event) =>
          updateParentDimension("operator", event.target.value)
        }
      >
        {DIMENSION_OPERATORS.filter(
          (operator) => operator.value !== "<" && operator.value !== "",
        ).map((operator) => (
          <MenuItem key={operator.label} value={operator.value}>
            {operator.label}
          </MenuItem>
        ))}
      </TextField>
      <ValueComponent
        dimension={parentDimension}
        dimensionOptions={dimensionOptions}
        index={0}
        onChange={updateParentDimension}
        onKeyDown={onKeyDown}
      />
    </Box>
  );
}
