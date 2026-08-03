import { Box, MenuItem, TextField } from "@mui/material";

import { getValueOptions, getVisualLabel } from "./LaypersonQueryUtils.js";

export default function LaypersonDimensionValue({
  dimension,
  index,
  onChange,
  onKeyDown,
}) {
  if (!dimension.operator)
    return <Box sx={{ display: { xs: "none", sm: "block" } }} />;
  const valueOptions = getValueOptions(dimension.field);
  const isYear = dimension.field === "Time";
  const options = [
    ...(!valueOptions?.some((option) => option.value === dimension.value) &&
    dimension.value
      ? [{ value: dimension.value, label: dimension.value, color: null }]
      : []),
    ...(valueOptions || []),
  ];
  return (
    <TextField
      select={Boolean(valueOptions)}
      type={isYear ? "number" : undefined}
      label={index === 0 ? (isYear ? "Year" : "Value") : undefined}
      size="small"
      slotProps={{
        htmlInput: {
          "aria-label": isYear ? "Year" : "Value",
          ...(isYear ? { inputMode: "numeric", step: 1 } : {}),
        },
      }}
      value={dimension.value}
      onChange={(event) => onChange("value", event.target.value)}
      onKeyDown={onKeyDown}
    >
      {valueOptions &&
        options.map((option) => (
          <MenuItem key={option.value} value={option.value}>
            <Box
              aria-hidden="true"
              component="span"
              data-testid={`${option.value}-color`}
              sx={{
                bgcolor: option.color,
                border: 1,
                borderColor: "divider",
                borderRadius: "50%",
                height: 10,
                mr: 1,
                width: 10,
              }}
            />
            {getVisualLabel(option.label)}
          </MenuItem>
        ))}
    </TextField>
  );
}
