import { Box, Typography } from "@mui/material";

import { VISUAL_CONTENT_MAX_WIDTH_PX } from "../../../nonview/constants/APP.js";
import LaypersonDimensions from "./LaypersonDimensions.js";
import LaypersonQueryFields from "./LaypersonQueryFields.js";
import {
  getDimensionParts,
  getDimensionString,
  getVisualQueryParts,
} from "./LaypersonQueryUtils.js";

export default function LaypersonVisualQueryInput({
  value,
  onChange,
  onSubmit,
  queryOptions,
}) {
  const parts = getVisualQueryParts(value);
  const dimensions = getDimensionParts(parts.dimensions);
  const options = queryOptions || { entities: [], dimensionsByEntity: {} };
  const entities = options.entities.includes(parts.entity)
    ? options.entities
    : [parts.entity, ...options.entities].filter(Boolean);
  const dimensionOptions = options.dimensionsByEntity[parts.entity] || [];
  const updatePart = (name, nextValue) =>
    onChange(
      ["entity", "dimensions", "aggregate", "visual"]
        .map((partName) => (partName === name ? nextValue : parts[partName]))
        .join("/"),
    );
  const updateDimensions = (nextDimensions) =>
    updatePart("dimensions", nextDimensions.map(getDimensionString).join("+"));
  const submitOnEnter = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      onSubmit();
    }
  };
  return (
    <Box
      sx={{
        display: "grid",
        gap: 1.5,
        gridTemplateColumns: { xs: "1fr", md: "repeat(3, minmax(0, 1fr))" },
        maxWidth: VISUAL_CONTENT_MAX_WIDTH_PX,
      }}
    >
      <LaypersonQueryFields
        parts={parts}
        entityOptions={entities}
        onUpdate={updatePart}
      />
      <LaypersonDimensions
        dimensions={dimensions}
        dimensionOptions={dimensionOptions}
        onAdd={() => updatePart("dimensions", `${parts.dimensions}+`)}
        onChange={(index, name, nextValue) =>
          updateDimensions(
            dimensions.map((dimension, position) =>
              position === index
                ? { ...dimension, [name]: nextValue }
                : dimension,
            ),
          )
        }
        onKeyDown={submitOnEnter}
        onRemove={(index) =>
          updateDimensions(
            dimensions.filter((_dimension, position) => position !== index),
          )
        }
      />
      <Typography
        variant="caption"
        sx={{ color: "text.secondary", gridColumn: "1 / -1" }}
      >
        Press Enter in a text field to update the visualization.
      </Typography>
    </Box>
  );
}
