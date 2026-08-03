import {
  Box,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import { useState } from "react";

import LaypersonVisualQueryInput from "../moles/LaypersonVisualQueryInput.js";

export default function VisualQueryForm({ value, onChange, onSubmit }) {
  const [mode, setMode] = useState("layperson");

  function submit(event) {
    event?.preventDefault();
    onSubmit();
  }

  function submitExpertQuery(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <Box
      component="form"
      aria-label="Visual query form"
      onSubmit={submit}
      sx={{ mb: 2 }}
    >
      <ToggleButtonGroup
        exclusive
        size="small"
        value={mode}
        onChange={(_event, nextMode) => nextMode && setMode(nextMode)}
        aria-label="Query input mode"
        sx={{ mb: 1.5 }}
      >
        <ToggleButton value="layperson">Layperson Mode</ToggleButton>
        <ToggleButton value="expert">Expert Mode</ToggleButton>
      </ToggleButtonGroup>
      {mode === "layperson" ? (
        <LaypersonVisualQueryInput
          value={value}
          onChange={onChange}
          onSubmit={submit}
        />
      ) : (
        <TextField
          fullWidth
          multiline
          minRows={2}
          maxRows={6}
          size="small"
          label="Visual query"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={submitExpertQuery}
          helperText="Press Enter to update; use Shift+Enter for a new line"
          slotProps={{
            htmlInput: {
              autoComplete: "off",
              spellCheck: false,
            },
          }}
          sx={{
            "& .MuiInputBase-input": {
              fontFamily: "monospace",
              overflowWrap: "anywhere",
            },
          }}
        />
      )}
    </Box>
  );
}
