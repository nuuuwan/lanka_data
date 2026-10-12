import { Box, Typography } from "@mui/material";

import { VISUAL_QUERY_MAX_WIDTH_PX } from "../../nonview/constants/APP.js";
import { QUERY_WIZARD_EXAMPLES } from "../../nonview/constants/QueryWizard.js";
import styles from "./LaypersonVisualQueryInput.module.css";
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
  queryOptions,
}) {
  const parts = getVisualQueryParts(value);
  const dimensions = getDimensionParts(parts.dimensions);
  const options = queryOptions || { entities: [], dimensionsByEntity: {} };
  const entities = options.entities.includes(parts.entity)
    ? options.entities
    : [parts.entity, ...options.entities].filter(Boolean);
  const dimensionOptions = options.dimensionsByEntity[parts.entity] || [];
  const valueOptionsByField = options.valuesByField || {};
  const updatePart = (name, nextValue) =>
    onChange(
      ["entity", "dimensions", "aggregate", "visual"]
        .map((partName) => (partName === name ? nextValue : parts[partName]))
        .join("/"),
    );
  const updateDimensions = (nextDimensions) =>
    updatePart("dimensions", nextDimensions.map(getDimensionString).join("+"));
  const preventSubmitOnEnter = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
    }
  };
  return (
    <Box
      sx={{
        display: "grid",
        gap: 1.5,
        maxWidth: VISUAL_QUERY_MAX_WIDTH_PX,
        width: "100%",
      }}
    >
      <LaypersonQueryFields
        parts={parts}
        entityOptions={entities}
        onUpdate={updatePart}
      />
      <Typography component="p" className={styles.examples}>
        Facets: {QUERY_WIZARD_EXAMPLES.facets}
      </Typography>
      <LaypersonDimensions
        dimensions={dimensions}
        dimensionOptions={dimensionOptions}
        valueOptionsByField={valueOptionsByField}
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
        onKeyDown={preventSubmitOnEnter}
        onRemove={(index) =>
          updateDimensions(
            dimensions.filter((_dimension, position) => position !== index),
          )
        }
      />
    </Box>
  );
}
