import { ListSubheader, MenuItem, TextField } from "@mui/material";

import {
  DIMENSION_OPERATORS,
  getFieldGroups,
} from "../../nonview/constants/VisualQueryOptions.js";
import { getVisualLabel } from "./LaypersonQueryUtils.js";

export function DimensionFieldSelect({ index, onChange, options, value }) {
  return (
    <TextField
      select
      label={index === 0 ? "Field" : undefined}
      size="small"
      slotProps={{ htmlInput: { "aria-label": "Field" } }}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    >
      {getFieldGroups(options).flatMap((group) => [
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
  );
}

export function DimensionOperatorSelect({ index, onChange, value }) {
  return (
    <TextField
      select
      label={index === 0 ? "Operator" : undefined}
      size="small"
      slotProps={{
        htmlInput: { "aria-label": "Operator" },
        inputLabel: { shrink: true },
        select: { displayEmpty: true },
      }}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    >
      {DIMENSION_OPERATORS.map((operator) => (
        <MenuItem key={operator.label} value={operator.value}>
          {operator.label}
        </MenuItem>
      ))}
    </TextField>
  );
}
