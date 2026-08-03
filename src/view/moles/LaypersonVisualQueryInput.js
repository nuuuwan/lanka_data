import { Box, MenuItem, TextField, Typography } from "@mui/material";

import VisualFactory from "./visuals/VisualFactory.js";

function getVisualQueryParts(visualQueryStr) {
  const [entity = "", dimensions = "", aggregate = "", visual = ""] =
    visualQueryStr.split("/");
  return { entity, dimensions, aggregate, visual };
}

function getVisualLabel(visual) {
  return visual.replace(/([a-z])([A-Z])/g, "$1 $2");
}

export default function LaypersonVisualQueryInput({
  value,
  onChange,
  onSubmit,
}) {
  const parts = getVisualQueryParts(value);

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

  return (
    <Box
      sx={{
        display: "grid",
        gap: 1.5,
        gridTemplateColumns: {
          xs: "1fr",
          md: "minmax(8rem, 1fr) minmax(16rem, 3fr) minmax(8rem, 1fr) minmax(10rem, 1fr)",
        },
      }}
    >
      <TextField
        label="What data?"
        size="small"
        value={parts.entity}
        onChange={(event) => updatePart("entity", event.target.value)}
        onKeyDown={submitOnEnter}
        helperText="For example, Person or Vote"
      />
      <TextField
        label="Group or filter by"
        size="small"
        value={parts.dimensions}
        onChange={(event) => updatePart("dimensions", event.target.value)}
        onKeyDown={submitOnEnter}
        helperText="Use + between fields; = adds a filter"
      />
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
        {VisualFactory.list().map((visual) => (
          <MenuItem key={visual} value={visual}>
            {getVisualLabel(visual)}
          </MenuItem>
        ))}
      </TextField>
      <Typography
        variant="caption"
        sx={{ color: "text.secondary", gridColumn: "1 / -1" }}
      >
        Press Enter in a text field to update the visualization.
      </Typography>
    </Box>
  );
}
