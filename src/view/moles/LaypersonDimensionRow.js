import CloseIcon from "@mui/icons-material/Close";
import { Box, IconButton } from "@mui/material";

import {
  DimensionFieldSelect,
  DimensionOperatorSelect,
} from "./DimensionSelectors.js";
import LaypersonDimensionValue from "./LaypersonDimensionValue.js";

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
      <DimensionFieldSelect
        index={index}
        options={fieldOptions}
        value={dimension.field}
        onChange={(value) => onChange("field", value)}
      />
      <DimensionOperatorSelect
        index={index}
        value={dimension.operator}
        onChange={(value) => onChange("operator", value)}
      />
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
        <CloseIcon />
      </IconButton>
    </Box>
  );
}
