import { Box, MenuItem, TextField } from "@mui/material";

import { getValueOptions, getVisualLabel } from "./LaypersonQueryUtils.js";

export default function BasicDimensionValue({
  dimension,
  index,
  onChange,
  onKeyDown,
}) {
  const valueOptions =
    dimension.operator && dimension.operator !== "<"
      ? getValueOptions(dimension.field)
      : null;
  const isYear = dimension.field === "Time";
  const selectedValues = dimension.value
    ? dimension.value.split(",").filter(Boolean)
    : [];
  const options = [
    ...(valueOptions
      ? selectedValues
          .filter(
            (value) => !valueOptions.some((option) => option.value === value),
          )
          .map((value) => ({ value, label: value, color: null }))
      : []),
    ...(valueOptions || []),
  ];
  return (
    <TextField
      select={Boolean(valueOptions)}
      slotProps={{
        htmlInput: {
          "aria-label": isYear ? "Year" : "Value",
          ...(isYear ? { inputMode: "numeric" } : {}),
        },
        ...(valueOptions ? { select: { multiple: true } } : {}),
      }}
      type={undefined}
      label={index === 0 ? (isYear ? "Year" : "Value") : undefined}
      size="small"
      value={valueOptions ? selectedValues : dimension.value}
      onChange={(event) =>
        onChange(
          "value",
          Array.isArray(event.target.value)
            ? event.target.value.join(",")
            : event.target.value,
        )
      }
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
