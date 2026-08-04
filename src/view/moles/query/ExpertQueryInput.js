import { TextField } from "@mui/material";

import { VISUAL_CONTENT_MAX_WIDTH_PX } from "../../../nonview/constants/APP.js";

export default function ExpertQueryInput({ onChange, onSubmit, value }) {
  function submitOnEnter(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    }
  }

  return (
    <TextField
      multiline
      minRows={2}
      maxRows={6}
      size="small"
      label="Visual query"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      onKeyDown={submitOnEnter}
      helperText="Press Enter to update; use Shift+Enter for a new line"
      slotProps={{
        htmlInput: { autoComplete: "off", spellCheck: false },
      }}
      sx={{
        maxWidth: VISUAL_CONTENT_MAX_WIDTH_PX,
        width: "100%",
        "& .MuiInputBase-input": { overflowWrap: "anywhere" },
      }}
    />
  );
}
