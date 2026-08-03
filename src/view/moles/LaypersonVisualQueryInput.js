import AddIcon from "@mui/icons-material/Add";
import DeleteOutlinedIcon from "@mui/icons-material/DeleteOutlined";
import {
  Box,
  Button,
  IconButton,
  ListSubheader,
  MenuItem,
  TextField,
  Typography,
} from "@mui/material";

import {
  DIMENSION_OPERATORS,
  VISUAL_GROUPS,
} from "../../nonview/constants/VisualQueryOptions.js";
import CategoryConcept from "../../nonview/core/thing/concept/category_concept/CategoryConcept.js";
import ThingFactory from "../../nonview/core/thing/thing_factory/ThingFactory.js";
import VisualFactory from "./visuals/VisualFactory.js";

function getVisualQueryParts(visualQueryStr) {
  const [entity = "", dimensions = "", aggregate = "", visual = ""] =
    visualQueryStr.split("/");
  return { entity, dimensions, aggregate, visual };
}

function getVisualLabel(visual) {
  return visual.replaceAll("_", " ").replace(/([a-z])([A-Z])/g, "$1 $2");
}

function getDimensionParts(dimensions) {
  return dimensions.split("+").map((dimension) => {
    const operatorIndex = dimension.search(/[=<]/);
    if (operatorIndex === -1) {
      return { field: dimension, operator: "", value: "" };
    }
    return {
      field: dimension.slice(0, operatorIndex),
      operator: dimension[operatorIndex],
      value: dimension.slice(operatorIndex + 1),
    };
  });
}

function getDimensionString({ field, operator, value }) {
  return `${field}${operator}${operator ? value : ""}`;
}

function getValueOptions(field) {
  try {
    const ThingClass = ThingFactory.fromKey(field);
    if (!(ThingClass.prototype instanceof CategoryConcept)) {
      return null;
    }

    const colorMap = ThingClass.getColorMap();
    return ThingClass.validValues().map((value) => ({
      value,
      label: value,
      color: colorMap[value] || null,
    }));
  } catch {
    return null;
  }
}

export default function LaypersonVisualQueryInput({
  value,
  onChange,
  onSubmit,
  queryOptions,
}) {
  const parts = getVisualQueryParts(value);
  const dimensions = getDimensionParts(parts.dimensions);
  const availableQueryOptions = queryOptions || {
    entities: [],
    dimensionsByEntity: {},
  };
  const entityOptions = availableQueryOptions.entities.includes(parts.entity)
    ? availableQueryOptions.entities
    : [parts.entity, ...availableQueryOptions.entities].filter(Boolean);
  const dimensionOptions =
    availableQueryOptions.dimensionsByEntity[parts.entity] || [];

  function updatePart(name, nextValue) {
    onChange(
      ["entity", "dimensions", "aggregate", "visual"]
        .map((partName) => (partName === name ? nextValue : parts[partName]))
        .join("/"),
    );
  }

  function submitOnEnter(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      onSubmit();
    }
  }

  function updateDimension(index, name, nextValue) {
    const nextDimensions = dimensions.map((dimension, dimensionIndex) =>
      dimensionIndex === index
        ? { ...dimension, [name]: nextValue }
        : dimension,
    );
    updatePart("dimensions", nextDimensions.map(getDimensionString).join("+"));
  }

  function addDimension() {
    updatePart("dimensions", `${parts.dimensions}+`);
  }

  function removeDimension(index) {
    updatePart(
      "dimensions",
      dimensions
        .filter((_dimension, dimensionIndex) => dimensionIndex !== index)
        .map(getDimensionString)
        .join("+"),
    );
  }

  return (
    <Box
      sx={{
        display: "grid",
        gap: 1.5,
        gridTemplateColumns: { xs: "1fr", md: "repeat(3, minmax(0, 1fr))" },
      }}
    >
      <TextField
        select
        label="What data?"
        size="small"
        value={parts.entity}
        onChange={(event) => updatePart("entity", event.target.value)}
        helperText="Choose the type of data"
      >
        {entityOptions.map((entity) => (
          <MenuItem key={entity} value={entity}>
            {entity}
          </MenuItem>
        ))}
      </TextField>
      <TextField
        label="Calculate"
        size="small"
        value={parts.aggregate}
        onChange={(event) => updatePart("aggregate", event.target.value)}
        onKeyDown={submitOnEnter}
        helperText="For example, Count"
      />
      <TextField
        select
        label="Show as"
        size="small"
        value={parts.visual}
        onChange={(event) => updatePart("visual", event.target.value)}
        helperText="Choose a visual"
      >
        {VISUAL_GROUPS.flatMap((group) => [
          <ListSubheader key={group.label}>{group.label}</ListSubheader>,
          ...group.visuals
            .filter((visual) => VisualFactory.list().includes(visual))
            .map((visual) => (
              <MenuItem key={visual} value={visual}>
                {getVisualLabel(visual)}
              </MenuItem>
            )),
        ])}
      </TextField>
      <Box
        sx={{
          bgcolor: "info.light",
          borderLeft: 4,
          borderColor: "info.main",
          borderRadius: 1,
          gridColumn: "1 / -1",
          p: 1.5,
        }}
      >
        <Typography component="div" variant="subtitle2" sx={{ mb: 1 }}>
          Group or filter
        </Typography>
        {dimensions.map((dimension, index) => (
          <Box
            key={index}
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
            <TextField
              select
              label={index === 0 ? "Field" : undefined}
              size="small"
              slotProps={{ htmlInput: { "aria-label": "Field" } }}
              value={dimension.field}
              onChange={(event) =>
                updateDimension(index, "field", event.target.value)
              }
            >
              {!dimensionOptions.includes(dimension.field) &&
                dimension.field && (
                  <MenuItem value={dimension.field}>{dimension.field}</MenuItem>
                )}
              {dimensionOptions.map((dimensionName) => (
                <MenuItem key={dimensionName} value={dimensionName}>
                  {getVisualLabel(dimensionName)}
                </MenuItem>
              ))}
            </TextField>
            <TextField
              select
              label={index === 0 ? "Operator" : undefined}
              size="small"
              slotProps={{
                htmlInput: { "aria-label": "Operator" },
                inputLabel: { shrink: true },
                select: { displayEmpty: true },
              }}
              value={dimension.operator}
              onChange={(event) =>
                updateDimension(index, "operator", event.target.value)
              }
            >
              {DIMENSION_OPERATORS.map((operator) => (
                <MenuItem key={operator.label} value={operator.value}>
                  {operator.label}
                </MenuItem>
              ))}
            </TextField>
            {dimension.operator ? (
              (() => {
                const valueOptions = getValueOptions(dimension.field);
                return (
                  <TextField
                    select={Boolean(valueOptions)}
                    label={index === 0 ? "Value" : undefined}
                    size="small"
                    slotProps={{ htmlInput: { "aria-label": "Value" } }}
                    value={dimension.value}
                    onChange={(event) =>
                      updateDimension(index, "value", event.target.value)
                    }
                    onKeyDown={submitOnEnter}
                  >
                    {valueOptions &&
                      [
                        ...(!valueOptions.some(
                          (option) => option.value === dimension.value,
                        ) && dimension.value
                          ? [
                              {
                                value: dimension.value,
                                label: dimension.value,
                                color: null,
                              },
                            ]
                          : []),
                        ...valueOptions,
                      ].map((option) => (
                        <MenuItem key={option.value} value={option.value}>
                          {option.color && (
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
                          )}
                          {getVisualLabel(option.label)}
                        </MenuItem>
                      ))}
                  </TextField>
                );
              })()
            ) : (
              <Box sx={{ display: { xs: "none", sm: "block" } }} />
            )}
            <IconButton
              aria-label={`Remove condition ${index + 1}`}
              disabled={dimensions.length === 1}
              onClick={() => removeDimension(index)}
              size="small"
            >
              <DeleteOutlinedIcon fontSize="small" />
            </IconButton>
          </Box>
        ))}
        <Button
          disabled={!dimensions.at(-1).field}
          onClick={addDimension}
          size="small"
          startIcon={<AddIcon />}
        >
          AND
        </Button>
      </Box>
      <Typography
        variant="caption"
        sx={{ color: "text.secondary", gridColumn: "1 / -1" }}
      >
        Press Enter in a text field to update the visualization.
      </Typography>
    </Box>
  );
}
