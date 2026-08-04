import { Box, ListSubheader, MenuItem, TextField } from "@mui/material";
import { useMemo } from "react";

import {
  DIMENSION_OPERATORS,
  getFieldGroups,
} from "../../nonview/constants/VisualQueryOptions.js";
import {
  getDimensionParts,
  getDimensionString,
  getValueOptions,
  getVisualLabel,
} from "./LaypersonQueryUtils.js";

export default function LaypersonDimensionValue({
  dimension,
  dimensionOptions,
  index,
  onChange,
  onKeyDown,
}) {
  const valueOptions = useMemo(
    () =>
      dimension.operator && dimension.operator !== "<"
        ? getValueOptions(dimension.field)
        : null,
    [dimension.field, dimension.operator],
  );
  if (!dimension.operator)
    return <Box sx={{ display: { xs: "none", sm: "block" } }} />;
  if (dimension.operator === "<") {
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
      <Box
        sx={{
          display: "grid",
          gap: 0.75,
          gridTemplateColumns: {
            xs: "1fr",
            md: "minmax(8rem, 1fr) minmax(5rem, auto) minmax(8rem, 1fr)",
          },
        }}
      >
        <TextField
          select
          label="Parent field"
          size="small"
          slotProps={{ htmlInput: { "aria-label": "Parent field" } }}
          value={parentDimension.field}
          onChange={(event) =>
            updateParentDimension("field", event.target.value)
          }
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
        <LaypersonDimensionValue
          dimension={parentDimension}
          dimensionOptions={dimensionOptions}
          index={0}
          onChange={updateParentDimension}
          onKeyDown={onKeyDown}
        />
      </Box>
    );
  }
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
