import { Box } from "@mui/material";
import { useMemo } from "react";

import DimensionValueInput from "./DimensionValueInput.js";
import { getValueOptions } from "./LaypersonQueryUtils.js";
import ParentDimensionValue from "./ParentDimensionValue.js";

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
  if (!dimension.operator) {
    return <Box sx={{ display: { xs: "none", sm: "block" } }} />;
  }
  if (dimension.operator === "<") {
    return (
      <ParentDimensionValue
        ValueComponent={LaypersonDimensionValue}
        dimension={dimension}
        dimensionOptions={dimensionOptions}
        onChange={onChange}
        onKeyDown={onKeyDown}
      />
    );
  }
  return (
    <DimensionValueInput
      dimension={dimension}
      index={index}
      onChange={onChange}
      onKeyDown={onKeyDown}
      valueOptions={valueOptions}
    />
  );
}
