import { Tab, Tabs, TextField } from "@mui/material";

import { VISUAL_CONTENT_MAX_WIDTH_PX } from "../../nonview/constants/APP.js";
import LaypersonVisualQueryInput from "./LaypersonVisualQueryInput.js";

export default function VisualQueryFields({
  mode,
  onModeChange,
  onChange,
  onExpertKeyDown,
  onSubmit,
  queryOptions,
  value,
}) {
  return (
    <>
      <Tabs
        value={mode}
        onChange={(_event, nextMode) => onModeChange(nextMode)}
        aria-label="Query input mode"
        sx={{
          borderBottom: 1,
          borderColor: "divider",
          mb: 2,
          minHeight: 40,
          "& .MuiTab-root": {
            minHeight: 40,
            px: 2.5,
            textTransform: "none",
          },
          "& .Mui-selected": { color: "text.primary", fontWeight: 700 },
        }}
      >
        <Tab label="Expert" value="expert" />
        <Tab label="Layperson" value="layperson" />
      </Tabs>
      {mode === "layperson" ? (
        <LaypersonVisualQueryInput
          value={value}
          onChange={onChange}
          onSubmit={onSubmit}
          queryOptions={queryOptions}
        />
      ) : (
        <TextField
          multiline
          minRows={2}
          maxRows={6}
          size="small"
          label="Visual query"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={onExpertKeyDown}
          helperText="Press Enter to update; use Shift+Enter for a new line"
          slotProps={{ htmlInput: { autoComplete: "off", spellCheck: false } }}
          sx={{
            maxWidth: VISUAL_CONTENT_MAX_WIDTH_PX,
            width: "100%",
            "& .MuiInputBase-input": { overflowWrap: "anywhere" },
          }}
        />
      )}
    </>
  );
}
